import os
import sys
import argparse
import copy
import time
import json
import random
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold

# Ensure paths resolve cleanly
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
for p in [BASE_DIR, SRC_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from preprocessing import prepare_official_split
from adasyn_balancer import AdaptiveClassBalancer
from model import get_model
from utils import compute_metrics, save_experiment_results, print_paper_comparison_table


# =====================================================================
# REPRODUCIBILITY CONTROLS
# =====================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =====================================================================
# DYNAMIC THRESHOLD CALIBRATION (TAU & THETA FROM VALIDATION SET)
# =====================================================================
def calibrate_dual_engine_thresholds(model, val_loader, normal_idx, device, percentile_theta=5, k_sigma=3.0):
    model.eval()
    benign_recon_losses = []
    correct_p_maxes = []

    with torch.no_grad():
        for X_val_b, y_val_b in val_loader:
            X_val_b = X_val_b.to(device)
            y_val_b = y_val_b.to(device)

            # 1. Reconstruction error on benign validation samples
            benign_mask = (y_val_b == normal_idx)
            if benign_mask.sum() > 0:
                recon_loss, _ = model.engine1_zero_day_guard.compute_anomaly_score(X_val_b[benign_mask])
                benign_recon_losses.extend(recon_loss.cpu().numpy().tolist())

            # 2. Maximum softmax confidence on correctly classified validation samples
            logits = model.engine2_classifier(X_val_b)
            probs = F.softmax(logits, dim=-1)
            p_max, preds = torch.max(probs, dim=-1)
            correct_mask = (preds == y_val_b)
            if correct_mask.sum() > 0:
                correct_p_maxes.extend(p_max[correct_mask].cpu().numpy().tolist())

    # Dynamically tune Tau (Reconstruction threshold)
    if len(benign_recon_losses) > 0:
        mu_recon = float(np.mean(benign_recon_losses))
        std_recon = float(np.std(benign_recon_losses))
        calibrated_tau = float(mu_recon + k_sigma * std_recon)
    else:
        mu_recon, std_recon = 0.01, 0.003
        calibrated_tau = 0.0191

    # Dynamically tune Theta (Confidence floor)
    if len(correct_p_maxes) > 0:
        calibrated_theta = float(np.percentile(correct_p_maxes, percentile_theta))
    else:
        calibrated_theta = 0.85

    model.tau_threshold = calibrated_tau
    model.theta_threshold = calibrated_theta

    print(f"\n  [+] Calibrated Engine 1 Tau Threshold   : {calibrated_tau:.6f} (mu={mu_recon:.6f}, std={std_recon:.6f})")
    print(f"  [+] Calibrated Engine 2 Theta Threshold : {calibrated_theta:.4f} ({percentile_theta}th percentile)")
    return calibrated_tau, calibrated_theta


# =====================================================================
# ENGINE 1 (AUTOENCODER) BENIGN PRE-TRAINING
# =====================================================================
def pretrain_engine1(model, X_train, y_train, normal_idx, device, batch_size=64, lr=0.001, epochs=5):
    print("  [*] Pretraining Engine 1 Autoencoder on benign samples...")
    criterion_recon = nn.MSELoss()
    ae_optimizer = optim.Adam(model.engine1_zero_day_guard.parameters(), lr=lr)
    
    benign_mask = (y_train == normal_idx)
    X_train_benign = X_train[benign_mask]
    benign_loader = DataLoader(
        TensorDataset(torch.tensor(X_train_benign, dtype=torch.float32)),
        batch_size=batch_size, shuffle=True
    )

    for ae_ep in range(1, epochs + 1):
        model.engine1_zero_day_guard.train()
        running_loss = 0.0
        for (x_b,) in benign_loader:
            x_b = x_b.to(device)
            ae_optimizer.zero_grad()
            recon, _ = model.engine1_zero_day_guard(x_b)
            loss_ae = criterion_recon(recon, x_b)
            loss_ae.backward()
            ae_optimizer.step()
            running_loss += loss_ae.item() * x_b.size(0)
        
        if ae_ep % 2 == 0 or ae_ep == epochs:
            avg_loss = running_loss / len(X_train_benign)
            print(f"      AE Epoch [{ae_ep}/{epochs}] | Benign Recon MSE: {avg_loss:.6f}")


# =====================================================================
# 1. THREE-WAY SPLIT PIPELINE (TRAIN / VAL / TEST)
# =====================================================================
def run_single_experiment(args, device):
    set_seed(args.seed)
    balance_suffix = f"{args.balancer}_balanced" if args.balance else "raw"
    ds_prefix = args.dataset_name.lower().replace("-", "_")
    hide_suffix = f"_loco_{args.hide_class.lower()}" if args.hide_class else ""
    exp_name = f"{ds_prefix}_{args.model}{hide_suffix}_{balance_suffix}_{args.epochs}ep"
    
    print("\n==================================================")
    print(f"    RUNNING EXPERIMENT: {exp_name.upper()}")
    print("==================================================")

    raw_df = pd.read_csv(args.dataset)

    # Truly unseen LOCO isolation: hide target attack entirely from train and val
    zero_day_df = None
    if args.hide_class:
        zero_day_df = raw_df[raw_df[args.target_col].astype(str).str.lower() == args.hide_class.lower()].copy()
        raw_df = raw_df[raw_df[args.target_col].astype(str).str.lower() != args.hide_class.lower()].copy()
        print(f"  [*] Zero-Day Class '{args.hide_class}' isolated: {len(zero_day_df)} samples reserved strictly for unseen evaluation.")

    # 3-Way Stratified Partition: 70% Train, 15% Validation, 15% Unseen Test
    stratify_col = raw_df[args.target_col] if args.target_col in raw_df else None
    train_df, temp_df = train_test_split(raw_df, test_size=0.30, random_state=args.seed, stratify=stratify_col)
    temp_stratify = temp_df[args.target_col] if args.target_col in temp_df else None
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=args.seed, stratify=temp_stratify)

    # Balancing applied ONLY on Training Partition
    if args.balance:
        print(f"\n--- STAGE: Minority Class Balancing via {args.balancer.upper()} (Training Partition Only) ---")
        balancer = AdaptiveClassBalancer(
            target_col=args.target_col, 
            random_state=args.seed,
            preferred_sampler=args.balancer
        )
        train_df = balancer.balance_dataset(train_df)

    print("\n--- STAGE: Feature Preprocessing & Normalization ---")
    X_train, X_val, y_train, y_val, preprocessor, _ = prepare_official_split(
        train_df, val_df, target_col=args.target_col, hide_class=None
    )
    X_test, y_test = preprocessor.transform(test_df)

    num_features = X_train.shape[1]
    class_names = [str(c) for c in preprocessor.target_encoder.classes_]
    num_classes = len(class_names)
    
    normal_idx = 0
    for idx, c in enumerate(class_names):
        if c.lower() in ['normal', 'benign']:
            normal_idx = idx
            break

    print(f"Features: {num_features} | Classes: {class_names} | Benign Index: {normal_idx}")
    print(f"Splits -> Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
        batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)),
        batch_size=args.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)),
        batch_size=args.batch_size, shuffle=False
    )

    model = get_model(args.model, input_features=num_features, num_classes=num_classes).to(device)
    criterion_cls = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    if args.model.lower() in ['dual-engine', 'dual_engine', 'dual']:
        pretrain_engine1(model, X_train, y_train, normal_idx, device, batch_size=args.batch_size, lr=args.lr, epochs=10)

    # Supervised Training Loop
    print(f"\n--- STAGE: Training Classifier for {args.epochs} Epochs ---")
    train_accs, train_losses = [], []
    val_accs, val_losses = [], []
    best_val_acc = -1.0
    best_model_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model.engine2_classifier(X_batch) if args.model.lower() in ['dual-engine', 'dual_engine', 'dual'] else model(X_batch)
            loss = criterion_cls(logits, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)
            _, preds = torch.max(logits, 1)
            total += y_batch.size(0)
            correct += (preds == y_batch).sum().item()

        scheduler.step()
        train_losses.append(running_loss / total)
        train_accs.append(correct / total)

        # Validation per Epoch
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for X_v, y_v in val_loader:
                X_v, y_v = X_v.to(device), y_v.to(device)
                v_logits = model.engine2_classifier(X_v) if args.model.lower() in ['dual-engine', 'dual_engine', 'dual'] else model(X_v)
                loss = criterion_cls(v_logits, y_v)
                v_loss += loss.item() * X_v.size(0)
                _, p = torch.max(v_logits, 1)
                v_total += y_v.size(0)
                v_correct += (p == y_v).sum().item()

        val_acc = v_correct / v_total
        val_losses.append(v_loss / v_total)
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())

        if epoch % max(1, args.epochs // 10) == 0 or epoch == args.epochs:
            print(f"Epoch [{epoch}/{args.epochs}] | Train Acc: {epoch_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

    last_model_state = copy.deepcopy(model.state_dict())

    # RESTORE BEST MODEL STATE FOR FINAL UNSEEN TESTING
    print("\n--- STAGE: Restoring Best Checkpoint for Final Unseen Test Evaluation ---")
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Dynamically calibrate Tau & Theta using validation data before final inference
    if args.model.lower() in ['dual-engine', 'dual_engine', 'dual']:
        calibrate_dual_engine_thresholds(model, val_loader, normal_idx, device)

    model.eval()
    y_preds, y_trues = [], []
    start_eval = time.time()
    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b = X_b.to(device)
            logits = model.engine2_classifier(X_b) if args.model.lower() in ['dual-engine', 'dual_engine', 'dual'] else model(X_b)
            _, p = torch.max(logits, 1)
            y_preds.extend(p.cpu().numpy())
            y_trues.extend(y_b.numpy())

    eval_time = time.time() - start_eval
    num_samples = len(y_trues)
    latency_ms = (eval_time / num_samples) * 1000 if num_samples > 0 else 0.0
    throughput = num_samples / eval_time if eval_time > 0 else 0.0

    metrics = compute_metrics(y_trues, y_preds, normal_idx=normal_idx, num_classes=num_classes)
    metrics['train_accuracy'] = train_accs[-1] if train_accs else 0.0
    metrics['best_val_accuracy'] = best_val_acc
    metrics['test_accuracy'] = metrics['accuracy']
    metrics['inference_latency_ms'] = latency_ms
    metrics['throughput_pps'] = throughput

    print("\n==================================================")
    print(f"  [+] BEST VAL ACCURACY      : {best_val_acc * 100:.2f}%")
    print(f"  [+] FINAL TEST ACCURACY    : {metrics['test_accuracy'] * 100:.2f}%")
    print(f"  [*] TEST MACRO F1-SCORE    : {metrics['f1_macro'] * 100:.2f}%")
    print(f"  [*] DETECTION RATE (DR)   : {metrics['detection_rate'] * 100:.2f}%")
    print(f"  [*] FALSE ALARM RATE (FAR) : {metrics['false_alarm_rate'] * 100:.2f}%")
    print("==================================================\n")

    # Evaluate Truly Unseen Zero-Day LOCO Class
    if zero_day_df is not None and len(zero_day_df) > 0:
        print(f"--- EVALUATING UNSEEN ZERO-DAY ATTACK: '{args.hide_class}' ({len(zero_day_df)} samples) ---")
        X_zd, _ = preprocessor.transform(zero_day_df)
        X_zd_tensor = torch.tensor(X_zd, dtype=torch.float32).to(device)
        with torch.no_grad():
            if args.model.lower() in ['dual-engine', 'dual_engine', 'dual']:
                verdicts, preds, p_max, recon_loss = model.predict_verdict(X_zd_tensor, normal_class_idx=normal_idx)
                detected = (verdicts == 2).sum().item()
                pct = (detected / len(X_zd_tensor)) * 100
                print(f"  Zero-Day Isolation Accuracy: {detected}/{len(X_zd_tensor)} ({pct:.2f}%)")
                print(f"  Mean Recon Loss: {recon_loss.mean().item():.4f} (tau={model.tau_threshold:.6f})")
                print(f"  Mean Softmax Confidence: {p_max.mean().item():.4f} (theta={model.theta_threshold:.4f})")

    # Save Preprocessor Artifacts alongside Checkpoints
    save_dir = os.path.join("results", exp_name)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "preprocessor.pkl"), "wb") as f:
        pickle.dump(preprocessor, f)

    save_experiment_results(
        exp_name, metrics, y_trues, y_preds, train_accs, train_losses,
        class_names, val_accs=val_accs, val_losses=val_losses,
        best_model_state=best_model_state, last_model_state=last_model_state
    )

    mapping = {'cnn': 'CNN', 'resnet': 'ResNet', 'resnest': 'ResNeSt', 'resnet-gru': 'ResNet-GRU', 'proposed': 'Proposed', 'dual-engine': 'Proposed'}
    print_paper_comparison_table(args.dataset_name, mapping.get(args.model.lower(), 'Proposed'), f"{args.epochs}epoch", metrics)


# =====================================================================
# 2. STRATIFIED K-FOLD PIPELINE (WITH BEST-MODEL RESTORATION)
# =====================================================================
def run_kfold_experiment(args, device):
    set_seed(args.seed)
    balance_suffix = f"{args.balancer}_balanced" if args.balance else "raw"
    ds_prefix = args.dataset_name.lower().replace("-", "_")
    exp_dir_name = f"{ds_prefix}_{args.model}_{balance_suffix}_{args.kfold}fold_{args.epochs}ep"
    results_dir = os.path.join("results", exp_dir_name)
    os.makedirs(results_dir, exist_ok=True)

    print("\n=======================================================")
    print(f"   RUNNING {args.kfold}-FOLD CROSS VALIDATION: {exp_dir_name.upper()}")
    print("=======================================================")

    raw_df = pd.read_csv(args.dataset)
    y_raw = raw_df[args.target_col].copy()

    skf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(raw_df, y_raw), 1):
        print(f"\n>>>>>>>>>>>>>>>>>>>> FOLD [{fold}/{args.kfold}] <<<<<<<<<<<<<<<<<<<<")
        fold_dir = os.path.join(results_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        train_df = raw_df.iloc[train_idx].copy()
        val_df = raw_df.iloc[val_idx].copy()

        if args.balance:
            print(f"  [*] Applying {args.balancer.upper()} Balancing on Training Fold...")
            balancer = AdaptiveClassBalancer(
                target_col=args.target_col, 
                random_state=args.seed,
                preferred_sampler=args.balancer
            )
            train_df = balancer.balance_dataset(train_df)

        X_train, X_val, y_train, y_val, preprocessor, _ = prepare_official_split(
            train_df, val_df, target_col=args.target_col, hide_class=args.hide_class
        )

        num_features = X_train.shape[1]
        class_names = [str(c) for c in preprocessor.target_encoder.classes_]
        num_classes = len(class_names)
        
        normal_idx = 0
        for idx, c in enumerate(class_names):
            if c.lower() in ['normal', 'benign']:
                normal_idx = idx
                break

        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
            batch_size=args.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)),
            batch_size=args.batch_size, shuffle=False
        )

        model = get_model(args.model, input_features=num_features, num_classes=num_classes).to(device)
        criterion_cls = nn.CrossEntropyLoss(label_smoothing=0.05)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

        if args.model.lower() in ['dual-engine', 'dual_engine', 'dual']:
            pretrain_engine1(model, X_train, y_train, normal_idx, device, batch_size=args.batch_size, lr=args.lr, epochs=5)

        train_accs, train_losses = [], []
        val_accs, val_losses = [], []
        best_val_acc = -1.0
        best_model_state = None

        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                logits = model.engine2_classifier(X_b) if args.model.lower() in ['dual-engine', 'dual_engine', 'dual'] else model(X_b)
                loss = criterion_cls(logits, y_b)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * X_b.size(0)
                _, preds = torch.max(logits, 1)
                total += y_b.size(0)
                correct += (preds == y_b).sum().item()

            scheduler.step()
            train_losses.append(running_loss / total)
            train_accs.append(correct / total)

            model.eval()
            v_loss, v_correct, v_total = 0.0, 0, 0
            with torch.no_grad():
                for X_v, y_v in val_loader:
                    X_v, y_v = X_v.to(device), y_v.to(device)
                    v_logits = model.engine2_classifier(X_v) if args.model.lower() in ['dual-engine', 'dual_engine', 'dual'] else model(X_v)
                    loss = criterion_cls(v_logits, y_v)
                    v_loss += loss.item() * X_v.size(0)
                    _, p = torch.max(v_logits, 1)
                    v_total += y_v.size(0)
                    v_correct += (p == y_v).sum().item()

            current_val_acc = v_correct / v_total
            val_losses.append(v_loss / v_total)
            val_accs.append(current_val_acc)

            if current_val_acc > best_val_acc:
                best_val_acc = current_val_acc
                best_model_state = copy.deepcopy(model.state_dict())

            if epoch % max(1, args.epochs // 5) == 0 or epoch == args.epochs:
                print(f"  Epoch [{epoch}/{args.epochs}] | Train Acc: {train_accs[-1]*100:.2f}% | Val Acc: {current_val_acc*100:.2f}%")

        # RESTORE BEST MODEL STATE FOR FOLD EVALUATION
        print(f"  [*] Restoring Fold {fold} best checkpoint for evaluation...")
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Dynamic validation threshold tuning per fold
        if args.model.lower() in ['dual-engine', 'dual_engine', 'dual']:
            calibrate_dual_engine_thresholds(model, val_loader, normal_idx, device)

        model.eval()
        y_preds, y_trues = [], []
        with torch.no_grad():
            for X_v, y_v in val_loader:
                X_v = X_v.to(device)
                v_logits = model.engine2_classifier(X_v) if args.model.lower() in ['dual-engine', 'dual_engine', 'dual'] else model(X_v)
                _, p = torch.max(v_logits, 1)
                y_preds.extend(p.cpu().numpy())
                y_trues.extend(y_v.numpy())

        metrics = compute_metrics(y_trues, y_preds, normal_idx=normal_idx, num_classes=num_classes)
        metrics['fold'] = fold
        metrics['best_val_accuracy'] = best_val_acc
        metrics['final_val_accuracy'] = metrics['accuracy']
        fold_metrics.append(metrics)

        torch.save(best_model_state, os.path.join(fold_dir, "best_model.pt"))
        torch.save(model.state_dict(), os.path.join(fold_dir, "last_model.pt"))
        with open(os.path.join(fold_dir, "preprocessor.pkl"), "wb") as f:
            pickle.dump(preprocessor, f)

        print(f"  Fold {fold} Evaluated Accuracy: {metrics['accuracy']*100:.2f}% | DR: {metrics['detection_rate']*100:.2f}% | FAR: {metrics['false_alarm_rate']*100:.2f}%")

    accs = [m['accuracy'] * 100 for m in fold_metrics]
    f1s = [m['f1_macro'] * 100 for m in fold_metrics]
    drs = [m['detection_rate'] * 100 for m in fold_metrics]
    fars = [m['false_alarm_rate'] * 100 for m in fold_metrics]

    summary = {
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "mean_f1_macro": float(np.mean(f1s)),
        "std_f1_macro": float(np.std(f1s)),
        "mean_detection_rate": float(np.mean(drs)),
        "std_detection_rate": float(np.std(drs)),
        "mean_false_alarm_rate": float(np.mean(fars)),
        "std_false_alarm_rate": float(np.std(fars)),
        "folds": fold_metrics
    }

    with open(os.path.join(results_dir, "kfold_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    print("\n=======================================================")
    print(f"        {args.kfold}-FOLD CROSS VALIDATION SUMMARY       ")
    print("=======================================================")
    print(f"  ACCURACY    : {summary['mean_accuracy']:.2f}% (+/- {summary['std_accuracy']:.2f}%)")
    print(f"  MACRO F1    : {summary['mean_f1_macro']:.2f}% (+/- {summary['std_f1_macro']:.2f}%)")
    print(f"  DETECTION RT: {summary['mean_detection_rate']:.2f}% (+/- {summary['std_detection_rate']:.2f}%)")
    print(f"  FALSE ALARM : {summary['mean_false_alarm_rate']:.2f}% (+/- {summary['std_false_alarm_rate']:.2f}%)")
    print("=======================================================\n")


# =====================================================================
# ENTRY POINT
# =====================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Confidence-Aware Dual-Engine IoT NIDS")
    parser.add_argument('--dataset', type=str, default='data/UNSW_NB15_combined.csv')
    parser.add_argument('--dataset_name', type=str, default='UNSW-NB15', choices=['UNSW-NB15', 'CIC-IDS2018', 'CIC-IOT2023'])
    parser.add_argument('--model', type=str, default='dual-engine', choices=['proposed', 'dual-engine', 'cnn', 'resnet', 'resnest', 'resnet-gru'])
    parser.add_argument('--target_col', type=str, default='attack_cat')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--balance', action='store_true', help='Enable minority class balancing')
    parser.add_argument('--balancer', type=str, default='adasyn', choices=['adasyn', 'smotenc', 'random', 'auto'],
                        help='Balancer algorithm: adasyn, smotenc, random, or auto')
    parser.add_argument('--hide_class', type=str, default=None, help='Class to hide for zero-day LOCO test')
    parser.add_argument('--kfold', type=int, default=0, help='Set > 1 (e.g., 5) for K-Fold CV; 0 for 3-way split')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device}")

    if args.kfold > 1:
        run_kfold_experiment(args, device)
    else:
        run_single_experiment(args, device)