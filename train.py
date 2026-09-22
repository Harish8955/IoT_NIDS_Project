import os
import argparse
import copy
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

try:
    from preprocessing import prepare_official_split
    from adasyn_balancer import ADASYNBalancer
    from model import get_model
    from utils import compute_metrics, save_experiment_results, print_paper_comparison_table
except ModuleNotFoundError:
    from src.preprocessing import prepare_official_split
    from src.adasyn_balancer import ADASYNBalancer
    from src.model import get_model
    from src.utils import compute_metrics, save_experiment_results, print_paper_comparison_table


def train_and_evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device}")

    balance_suffix = "adasyn_balanced" if args.balance else "raw"
    ds_prefix = args.dataset_name.lower().replace("-", "_")
    hide_suffix = f"_loco_{args.hide_class.lower()}" if args.hide_class else ""
    exp_name = f"{ds_prefix}_{args.model}{hide_suffix}_{balance_suffix}_{args.epochs}ep"
    print(f"\n==================================================")
    print(f"    RUNNING EXPERIMENT: {exp_name.upper()}")
    print(f"==================================================")

    # 1. Load Data
    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Dataset path '{args.dataset}' not found.")
    
    raw_df = pd.read_csv(args.dataset)
    
    # 2. Strict Train/Test Separation (Prevent Balancing Data Leakage)[cite: 7, 8]
    if args.test_dataset and os.path.exists(args.test_dataset):
        train_raw_df = raw_df
        test_raw_df = pd.read_csv(args.test_dataset)
    else:
        stratify_col = raw_df[args.target_col] if args.target_col in raw_df else None
        train_raw_df, test_raw_df = train_test_split(
            raw_df, test_size=args.test_size, random_state=42, stratify=stratify_col
        )

    # 3. ADASYN Oversampling on Training Set Only[cite: 7, 8]
    if args.balance:
        print("\n--- STAGE: ADASYN Minority Class Balancing (Training Set Only) ---")
        balancer = ADASYNBalancer(target_col=args.target_col)
        train_raw_df = balancer.balance_dataset(train_raw_df)

    # 4. Preprocessing & Min-Max Normalization[cite: 7, 8]
    print("\n--- STAGE: Feature Preprocessing & Normalization ---")
    X_train, X_test, y_train, y_test, preprocessor, zero_day_df = prepare_official_split(
        train_raw_df, test_raw_df, target_col=args.target_col, hide_class=args.hide_class
    )

    num_features = X_train.shape[1]
    num_classes = len(preprocessor.target_encoder.classes_)
    class_names = list(preprocessor.target_encoder.classes_)
    normal_idx = class_names.index('Normal') if 'Normal' in class_names else 0

    print(f"Features: {num_features} | Classes: {class_names}")
    print(f"Train samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}")

    # 5. DataLoaders
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
        batch_size=args.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)),
        batch_size=args.batch_size, shuffle=False
    )

    model = get_model(args.model, input_features=num_features, num_classes=num_classes).to(device)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_recon = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 6. Unsupervised Benign Pretraining for Engine 1 (Dual-Engine Mode)[cite: 7, 8]
    if args.model.lower() in ['dual-engine', 'dual_engine', 'dual']:
        print("\n--- STAGE: Engine 1 Autoencoder Unsupervised Training (Benign Traffic Only) ---")
        ae_optimizer = optim.Adam(model.engine1_zero_day_guard.parameters(), lr=args.lr)
        benign_mask = (y_train == normal_idx)
        X_train_benign = X_train[benign_mask]
        
        benign_loader = DataLoader(
            TensorDataset(torch.tensor(X_train_benign, dtype=torch.float32)),
            batch_size=args.batch_size, shuffle=True
        )
        
        for ae_ep in range(1, 11):
            model.engine1_zero_day_guard.train()
            running_ae_loss = 0.0
            for (x_b,) in benign_loader:
                x_b = x_b.to(device)
                ae_optimizer.zero_grad()
                recon, _ = model.engine1_zero_day_guard(x_b)
                loss_ae = criterion_recon(recon, x_b)
                loss_ae.backward()
                ae_optimizer.step()
                running_ae_loss += loss_ae.item() * x_b.size(0)
            avg_ae_loss = running_ae_loss / len(X_train_benign)
            if ae_ep % 2 == 0 or ae_ep == 10:
                print(f"AE Epoch [{ae_ep}/10] | Benign Reconstruction MSE: {avg_ae_loss:.6f}")

    # 7. Supervised Training Loop
    print(f"\n--- STAGE: Training Classifier for {args.epochs} Epochs ---")
    train_accs, train_losses = [], []
    test_accs, test_losses = [], []
    best_test_acc = -1.0
    best_model_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            
            if args.model.lower() in ['dual-engine', 'dual_engine', 'dual']:
                logits = model.engine2_classifier(X_batch)
            else:
                logits = model(X_batch)
                
            loss = criterion_cls(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * X_batch.size(0)
            _, preds = torch.max(logits, 1)
            total += y_batch.size(0)
            correct += (preds == y_batch).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        # Validation per Epoch
        model.eval()
        t_loss, t_correct, t_total = 0.0, 0, 0
        with torch.no_grad():
            for X_t, y_t in test_loader:
                X_t, y_t = X_t.to(device), y_t.to(device)
                if args.model.lower() in ['dual-engine', 'dual_engine', 'dual']:
                    t_logits = model.engine2_classifier(X_t)
                else:
                    t_logits = model(X_t)
                loss = criterion_cls(t_logits, y_t)
                t_loss += loss.item() * X_t.size(0)
                _, p = torch.max(t_logits, 1)
                t_total += y_t.size(0)
                t_correct += (p == y_t).sum().item()

        test_acc = t_correct / t_total
        test_losses.append(t_loss / t_total)
        test_accs.append(test_acc)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_state = copy.deepcopy(model.state_dict())

        if epoch % max(1, args.epochs // 10) == 0 or epoch == args.epochs:
            print(f"Epoch [{epoch}/{args.epochs}] | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%")

    last_model_state = copy.deepcopy(model.state_dict())

    # 8. Test Set Evaluation
    model.eval()
    y_preds, y_trues = [], []
    start_eval = time.time()
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            if args.model.lower() in ['dual-engine', 'dual_engine', 'dual']:
                logits = model.engine2_classifier(X_batch)
            else:
                logits = model(X_batch)
            _, p = torch.max(logits, 1)
            y_preds.extend(p.cpu().numpy())
            y_trues.extend(y_batch.numpy())

    eval_time = time.time() - start_eval
    num_samples = len(y_trues)
    latency_ms = (eval_time / num_samples) * 1000 if num_samples > 0 else 0.0
    throughput = num_samples / eval_time if eval_time > 0 else 0.0

    metrics = compute_metrics(y_trues, y_preds)
    metrics['train_accuracy'] = train_accs[-1] if train_accs else 0.0
    metrics['test_accuracy'] = metrics['accuracy']
    metrics['inference_latency_ms'] = latency_ms
    metrics['throughput_pps'] = throughput

    print(f"\n==================================================")
    print(f"  [+] FINAL TEST ACCURACY    : {metrics['test_accuracy'] * 100:.2f}%")
    print(f"  [*] TEST MACRO F1-SCORE    : {metrics['f1_macro'] * 100:.2f}%")
    print(f"  [*] FALSE ALARM RATE (FAR) : {metrics['false_alarm_rate'] * 100:.2f}%")
    print(f"  [*] DETECTION RATE (DR)   : {metrics['detection_rate'] * 100:.2f}%")
    print(f"==================================================\n")

    # 9. Zero-Day LOCO Evaluation[cite: 7, 8]
    if zero_day_df is not None and len(zero_day_df) > 0:
        print(f"--- LOCO ZERO-DAY TEST: '{args.hide_class}' ({len(zero_day_df)} samples) ---")
        X_zd, _ = preprocessor.transform(zero_day_df)
        X_zd_tensor = torch.tensor(X_zd, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            if args.model.lower() in ['dual-engine', 'dual_engine', 'dual']:
                verdicts, preds, p_max, recon_loss = model.predict_verdict(X_zd_tensor, normal_class_idx=normal_idx)
                detected = (verdicts == 2).sum().item()
                pct = (detected / len(X_zd_tensor)) * 100
                print(f"  Zero-Day Detection Rate: {detected}/{len(X_zd_tensor)} ({pct:.2f}%)")
                print(f"  Mean Recon Loss: {recon_loss.mean().item():.4f} (tau={model.tau_threshold})")
                print(f"  Mean P_max: {p_max.mean().item():.4f} (theta={model.theta_threshold})")

    # 10. Save Results & Benchmark Compare
    save_experiment_results(
        exp_name, metrics, y_trues, y_preds, train_accs, train_losses,
        class_names, test_accs=test_accs, test_losses=test_losses,
        best_model_state=best_model_state, last_model_state=last_model_state
    )

    mapping = {'cnn': 'CNN', 'resnet': 'ResNet', 'resnest': 'ResNeSt', 'resnet-gru': 'ResNet-GRU', 'proposed': 'Proposed', 'dual-engine': 'Proposed'}
    print_paper_comparison_table(args.dataset_name, mapping.get(args.model.lower(), 'Proposed'), f"{args.epochs}epoch", metrics)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Confidence-Aware Dual-Engine IoT NIDS")
    parser.add_argument('--dataset', type=str, default='data/UNSW_NB15_combined.csv')
    parser.add_argument('--test_dataset', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default='UNSW-NB15', choices=['UNSW-NB15', 'CIC-IDS2018', 'CIC-IOT2023'])
    parser.add_argument('--model', type=str, default='dual-engine', choices=['proposed', 'dual-engine', 'cnn', 'resnet', 'resnest', 'resnet-gru'])
    parser.add_argument('--target_col', type=str, default='attack_cat')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--balance', action='store_true', help='Enable ADASYN minority class balancing')
    parser.add_argument('--hide_class', type=str, default=None)
    args = parser.parse_args()
    train_and_evaluate(args)