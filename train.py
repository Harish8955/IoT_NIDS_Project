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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, f1_score
from sklearn.preprocessing import label_binarize

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
for p in [BASE_DIR, SRC_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from preprocessing import TrafficDataPreprocessor
from adasyn_balancer import AdaptiveClassBalancer
from ctgan_balancer import CTGANBalancer
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


def clean_target_rows(df, target_col):
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' is missing from the dataset.")
    labels = df[target_col].astype('string').str.strip()
    invalid = labels.str.lower().isin(['nan', 'none', 'label']) | df[target_col].isna()
    return df.loc[~invalid].copy()


class FocalLoss(nn.Module):
    def __init__(self, class_weights=None, gamma=2.0):
        super().__init__()
        self.register_buffer('class_weights', class_weights)
        self.gamma = gamma

    def forward(self, logits, targets):
        log_pt = F.log_softmax(logits, dim=1).gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        loss = -((1.0 - pt) ** self.gamma) * log_pt
        if self.class_weights is not None:
            loss = loss * self.class_weights[targets]
        return loss.mean()


class ModelEMA:
    """Exponential moving average copy used for validation and best checkpoints."""
    def __init__(self, model, decay=0.99):
        self.module = copy.deepcopy(model).eval()
        self.decay = decay
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        source_state = model.state_dict()
        for name, ema_value in self.module.state_dict().items():
            source_value = source_state[name].detach()
            if torch.is_floating_point(ema_value):
                ema_value.mul_(self.decay).add_(source_value, alpha=1.0 - self.decay)
            else:
                ema_value.copy_(source_value)


def moving_average(values, window=3):
    values = list(values)
    return [float(np.mean(values[max(0, i - window + 1):i + 1])) for i in range(len(values))]


def make_classifier_loss(args, y_train, num_classes, device):
    weights = None
    if args.loss in ('weighted', 'focal'):
        counts = np.bincount(np.asarray(y_train, dtype=np.int64), minlength=num_classes)
        if np.any(counts == 0):
            raise ValueError('Every classifier class must have at least one training sample.')
        weights = len(y_train) / (num_classes * counts.astype(np.float64))
        weights = torch.tensor(weights, dtype=torch.float32, device=device)
        # Keep the average class weight at one to preserve the learning-rate scale.
        weights = weights / weights.mean()
    if args.loss == 'focal':
        return FocalLoss(class_weights=weights, gamma=args.focal_gamma)
    return nn.CrossEntropyLoss(
        weight=weights,
        label_smoothing=args.label_smoothing if args.loss == 'cross_entropy' else 0.0
    )


def prepare_training_arrays(train_df, val_df, args, seed, test_df=None):
    """Fit preprocessing on original train rows; resample only encoded train rows."""
    preprocessor = TrafficDataPreprocessor(target_col=args.target_col)
    X_train, y_train, _ = preprocessor.fit_transform(train_df)
    X_val, y_val = preprocessor.transform(val_df)
    X_test = y_test = None
    preprocessor.balancing_methods = {"all": "none (balancing disabled)"}
    if test_df is not None:
        X_test, y_test = preprocessor.transform(test_df)

    if args.balance:
        if args.balancer == 'ctgan':
            # CTGAN sees training rows only. Its samples use the original-train
            # encoder/scaler so validation/test never influence preprocessing.
            balanced_raw = CTGANBalancer(
                target_col=args.target_col, epochs=args.ctgan_epochs
            ).balance_dataset(train_df)
            X_train, y_train = preprocessor.transform(balanced_raw)
            preprocessor.balancing_methods = {"all": "CTGAN"}
        else:
            train_matrix = pd.DataFrame(X_train)
            train_matrix['__target__'] = np.asarray(y_train, dtype=np.int64)
            sampler = AdaptiveClassBalancer(
                target_col='__target__', max_minority_ratio=args.adasyn_ratio,
                random_state=seed, preferred_sampler=args.balancer
            )
            balanced = sampler.balance_dataset(train_matrix)
            X_train = balanced.drop(columns=['__target__']).to_numpy(dtype=np.float32)
            y_train = balanced['__target__'].to_numpy(dtype=np.int64)
            preprocessor.balancing_methods = sampler.actual_methods

    return X_train, X_val, y_train, y_val, X_test, y_test, preprocessor


def named_balancing_methods(preprocessor, class_names):
    """Map encoded class IDs in balancer diagnostics back to readable labels."""
    methods = getattr(preprocessor, 'balancing_methods', {})
    named = {}
    for class_id, method in methods.items():
        try:
            idx = int(class_id)
            label = class_names[idx] if 0 <= idx < len(class_names) else str(class_id)
        except (TypeError, ValueError):
            label = str(class_id)
        named[label] = method
    return named


def evaluate_classifier(model, loader, criterion, device):
    model.eval()
    total_loss, total = 0.0, 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            logits = model.engine2_classifier(X_b) if hasattr(model, 'engine2_classifier') else model(X_b)
            loss = criterion(logits, y_b)
            total_loss += loss.item() * X_b.size(0)
            total += X_b.size(0)
            y_true.extend(y_b.cpu().numpy())
            y_pred.extend(logits.argmax(dim=1).cpu().numpy())
    val_loss = total_loss / total if total else 0.0
    val_acc = float(np.mean(np.asarray(y_true) == np.asarray(y_pred))) if total else 0.0
    val_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0) if total else 0.0
    return val_loss, val_acc, val_f1


# =====================================================================
# PLOTTING AND ARTIFACT EXPORT HELPERS
# =====================================================================
def plot_learning_curves(history, save_path):
    epochs = history["epoch"]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 5))

    ax1.plot(epochs, history["train_loss"], 'b-', lw=1.8, label='Train loss (raw)')
    ax1.plot(epochs, history["val_loss"], color='crimson', lw=1.5, label='Validation loss (raw)')
    if "test_loss" in history:
        ax1.plot(epochs, history["test_loss"], color='darkorange', lw=1.5, label='Test loss (raw)')
    ax1.set_title('Loss', fontsize=12, pad=10)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend(fontsize=8)

    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], 'b-', lw=1.8, label='Train accuracy (raw)')
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], color='green', lw=1.5, label='Validation accuracy (raw)')
    if "test_acc" in history:
        ax2.plot(epochs, [a * 100 for a in history["test_acc"]], color='darkorange', lw=1.5, label='Test accuracy (raw)')
    ax2.set_title('Accuracy', fontsize=12, pad=10)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_ylim(0, 100)
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.legend(fontsize=8)

    if "val_macro_f1" in history:
        ax3.plot(epochs, [a * 100 for a in history["val_macro_f1"]], color='purple', lw=1.6, label='Validation macro-F1 (raw)')
    ax3.set_title('Validation Macro-F1', fontsize=12, pad=10)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Macro-F1 (%)')
    ax3.set_ylim(0, 100)
    ax3.grid(True, linestyle='--', alpha=0.4)
    ax3.legend(fontsize=8)

    fig.suptitle('Training, Validation, and Test Curves', fontsize=14, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.15)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

    plt.figure(figsize=(max(9, len(class_names) * 0.45), max(7, len(class_names) * 0.35)))
    sns.heatmap(
        cm_norm,
        annot=len(class_names) <= 15,
        fmt=".2f" if len(class_names) <= 15 else "",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True
    )
    plt.title("Normalized Confusion Matrix (Test Set)", fontsize=13, pad=12)
    plt.ylabel("True Label", fontsize=11)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_roc_curves(y_true, y_probs, class_names, save_path):
    num_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(num_classes)))
    plt.figure(figsize=(10, 8))

    for i in range(num_classes):
        if np.sum(y_bin[:, i]) > 0:
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
            roc_auc = auc(fpr, tpr)
            if num_classes <= 10 or roc_auc < 0.99 or i < 6:
                plt.plot(fpr, tpr, lw=1.5, label=f"{class_names[i]} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate", fontsize=11)
    plt.title("Multi-Class ROC Curves", fontsize=13)
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# =====================================================================
# CALIBRATION & PRETRAINING
# =====================================================================
def _select_proxy_class(class_names, y_train, y_val, normal_idx, requested='auto', hidden_class=None):
    """Select a known attack family to exclude from the proxy classifier."""
    train_counts = np.bincount(np.asarray(y_train, dtype=np.int64), minlength=len(class_names))
    val_counts = np.bincount(np.asarray(y_val, dtype=np.int64), minlength=len(class_names))
    excluded = (hidden_class or '').strip().casefold()
    candidates = [
        idx for idx, name in enumerate(class_names)
        if idx != normal_idx and name.strip().casefold() != excluded
        and train_counts[idx] > 0 and val_counts[idx] > 0
    ]
    if requested and requested.strip().casefold() != 'auto':
        matches = [idx for idx, name in enumerate(class_names)
                   if name.strip().casefold() == requested.strip().casefold()]
        if not matches or matches[0] not in candidates:
            available = [class_names[idx] for idx in candidates]
            raise ValueError(
                f"Proxy calibration class '{requested}' is unavailable. "
                f"Choose one of these known attack classes: {available}"
            )
        return matches[0]
    if not candidates:
        raise ValueError("Proxy zero-day calibration needs a known attack class in both train and validation data.")
    # Use the largest validation attack family so the proxy recall estimate is
    # less sensitive to a handful of samples. Worms (or --hide_class) is excluded.
    return max(candidates, key=lambda idx: (val_counts[idx], train_counts[idx], class_names[idx]))


def _make_proxy_loss(args, y_train, num_classes, device):
    """Build the configured loss while allowing the deliberately absent proxy class."""
    counts = np.bincount(np.asarray(y_train, dtype=np.int64), minlength=num_classes).astype(np.float64)
    active = counts > 0
    weights = None
    if args.loss in ('weighted', 'focal'):
        weights_np = np.zeros(num_classes, dtype=np.float32)
        weights_np[active] = len(y_train) / (active.sum() * counts[active])
        weights_np[active] /= weights_np[active].mean()
        weights = torch.tensor(weights_np, dtype=torch.float32, device=device)
    if args.loss == 'focal':
        return FocalLoss(class_weights=weights, gamma=args.focal_gamma)
    return nn.CrossEntropyLoss(
        weight=weights,
        label_smoothing=args.label_smoothing if args.loss == 'cross_entropy' else 0.0
    )


def _train_proxy_classifier(args, reference_model, X_train, y_train, X_val, y_val,
                            proxy_idx, num_classes, device):
    """Train a same-architecture calibration copy with the proxy family excluded."""
    train_keep = np.asarray(y_train) != proxy_idx
    val_keep = np.asarray(y_val) != proxy_idx
    X_proxy_train = np.asarray(X_train, dtype=np.float32)[train_keep]
    y_proxy_train = np.asarray(y_train, dtype=np.int64)[train_keep]
    X_known_val = np.asarray(X_val, dtype=np.float32)[val_keep]
    y_known_val = np.asarray(y_val, dtype=np.int64)[val_keep]
    X_proxy_val = np.asarray(X_val, dtype=np.float32)[~val_keep]
    if not len(X_proxy_train) or not len(X_proxy_val) or not len(X_known_val):
        raise ValueError("Proxy calibration requires proxy-family training and validation rows plus known validation rows.")

    counts = np.bincount(y_proxy_train, minlength=num_classes)
    if np.any(counts[np.arange(num_classes) != proxy_idx] == 0):
        missing = [i for i in range(num_classes) if i != proxy_idx and counts[i] == 0]
        raise ValueError(f"Proxy calibration is missing training samples for known class IDs {missing}.")

    proxy_model = get_model(args.model, input_features=X_train.shape[1], num_classes=num_classes).to(device)
    # Engine 1 is the benign-trained autoencoder from the final model. Only
    # Engine 2 needs a proxy copy because the proxy experiment asks whether an
    # unseen family is rejected by the classifier-confidence gate.
    proxy_model.engine1_zero_day_guard.load_state_dict(
        reference_model.engine1_zero_day_guard.state_dict()
    )
    proxy_model.engine1_zero_day_guard.eval()

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_proxy_train), torch.tensor(y_proxy_train)),
        batch_size=args.batch_size, shuffle=True,
        drop_last=len(X_proxy_train) >= args.batch_size
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_known_val), torch.tensor(y_known_val)),
        batch_size=args.batch_size, shuffle=False
    )
    criterion = _make_proxy_loss(args, y_proxy_train, num_classes, device)
    optimizer = optim.Adam(proxy_model.engine2_classifier.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
    )
    ema = ModelEMA(proxy_model, decay=args.ema_decay)
    best_f1 = -1.0
    best_state = None
    stale_epochs = 0
    val_window = []

    for epoch in range(1, args.epochs + 1):
        proxy_model.train()
        proxy_model.engine1_zero_day_guard.eval()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = proxy_model.engine2_classifier(X_b)
            loss = criterion(logits, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(proxy_model.engine2_classifier.parameters(), args.grad_clip)
            optimizer.step()
            ema.update(proxy_model)

        val_loss, _, _ = evaluate_classifier(ema.module, val_loader, criterion, device)
        ema.module.eval()
        known_truth, known_prediction = [], []
        with torch.no_grad():
            for X_b, y_b in val_loader:
                logits = ema.module.engine2_classifier(X_b.to(device))
                known_truth.extend(y_b.numpy().tolist())
                known_prediction.extend(logits.argmax(dim=1).cpu().numpy().tolist())
        known_label_ids = [idx for idx in range(num_classes) if idx != proxy_idx]
        val_f1 = f1_score(
            known_truth, known_prediction, labels=known_label_ids,
            average='macro', zero_division=0
        )
        scheduler.step(val_loss)
        val_window.append(val_f1)
        smoothed_f1 = float(np.mean(val_window[-args.smoothing_window:]))
        if smoothed_f1 > best_f1 + args.min_delta:
            best_f1 = smoothed_f1
            best_state = copy.deepcopy(ema.module.engine2_classifier.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("Proxy calibration classifier did not produce a selectable checkpoint.")
    proxy_model.engine2_classifier.load_state_dict(best_state)
    proxy_model.eval()
    return proxy_model, X_known_val, y_known_val, X_proxy_val, best_f1


def _collect_threshold_scores(engine1_model, classifier_model, X, device, batch_size=1024):
    recon_values, confidence_values, prediction_values = [], [], []
    X = torch.as_tensor(X, dtype=torch.float32)
    engine1_model.eval()
    classifier_model.eval()
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = X[start:start + batch_size].to(device)
            recon, _ = engine1_model.engine1_zero_day_guard.compute_anomaly_score(batch)
            logits = classifier_model.engine2_classifier(batch)
            probs = F.softmax(logits, dim=-1)
            confidence, prediction = probs.max(dim=-1)
            recon_values.extend(recon.cpu().numpy().tolist())
            confidence_values.extend(confidence.cpu().numpy().tolist())
            prediction_values.extend(prediction.cpu().numpy().tolist())
    return (np.asarray(recon_values, dtype=np.float64),
            np.asarray(confidence_values, dtype=np.float64),
            np.asarray(prediction_values, dtype=np.int64))


def calibrate_dual_engine_thresholds_with_proxy(
    model, X_train, y_train, X_val, y_val, class_names, normal_idx,
    args, device
):
    """Jointly sweep tau/theta using an attack family excluded from a proxy classifier."""
    proxy_idx = _select_proxy_class(
        class_names, y_train, y_val, normal_idx,
        requested=args.calibration_class, hidden_class=args.hide_class
    )
    print(f"  [*] Pseudo-zero-day calibration family: {class_names[proxy_idx]} (excluded from calibration classifier)")
    proxy_model, X_known, y_known, X_pseudo, proxy_val_f1 = _train_proxy_classifier(
        args, model, X_train, y_train, X_val, y_val, proxy_idx,
        len(class_names), device
    )
    known_recon, known_conf, known_pred = _collect_threshold_scores(
        model, proxy_model, X_known, device
    )
    pseudo_recon, pseudo_conf, pseudo_pred = _collect_threshold_scores(
        model, proxy_model, X_pseudo, device
    )

    benign_values = known_recon[y_known == normal_idx]
    if not len(benign_values):
        raise ValueError("Threshold calibration requires benign validation traffic.")
    legacy_tau = float(np.mean(benign_values) + 3.0 * np.std(benign_values))
    combined_recon = np.concatenate([known_recon, pseudo_recon])
    combined_conf = np.concatenate([known_conf, pseudo_conf])
    quantiles = np.linspace(0.0, 1.0, 51)
    tau_candidates = np.unique(np.concatenate([
        np.quantile(combined_recon, quantiles),
        np.asarray([legacy_tau, float(np.max(known_recon)), 0.0])
    ]))
    theta_candidates = np.unique(np.concatenate([
        np.quantile(combined_conf, quantiles), np.asarray([0.0, 1.0])
    ]))

    labels = np.unique(y_known)
    max_allowed = float(args.calibration_max_false_zero_day_rate)
    tau_candidates = np.sort(tau_candidates)
    best = None
    feasible_pair_count = 0
    known_by_class = {int(label): y_known == label for label in labels}
    for theta in np.sort(theta_candidates):
        known_trigger = (known_conf < theta) | (known_pred == normal_idx)
        pseudo_trigger = (pseudo_conf < theta) | (pseudo_pred == normal_idx)
        known_recon_sorted = {
            label: np.sort(known_recon[known_trigger & class_mask])
            for label, class_mask in known_by_class.items()
        }
        pseudo_recon_sorted = np.sort(pseudo_recon[pseudo_trigger])
        for tau in tau_candidates:
            per_class_false_rates = {}
            for label, sorted_recon in known_recon_sorted.items():
                class_size = int(known_by_class[label].sum())
                flagged = len(sorted_recon) - int(np.searchsorted(sorted_recon, tau, side='right'))
                per_class_false_rates[class_names[label]] = flagged / class_size * 100.0
            max_false_rate = max(per_class_false_rates.values(), default=0.0)
            if max_false_rate > max_allowed:
                continue
            feasible_pair_count += 1
            pseudo_flagged = len(pseudo_recon_sorted) - int(
                np.searchsorted(pseudo_recon_sorted, tau, side='right')
            )
            pseudo_recall = pseudo_flagged / len(pseudo_recon) * 100.0 if len(pseudo_recon) else 0.0
            # Prefer higher proxy recall, then fewer known false flags, and
            # finally the more conservative (higher tau/lower theta) pair.
            score = (pseudo_recall, -max_false_rate, float(tau), -float(theta))
            if best is None or score > best['score']:
                best = {
                    'score': score, 'tau': float(tau), 'theta': float(theta),
                    'pseudo_recall': pseudo_recall,
                    'known_false_rate': max_false_rate,
                    'per_class_false_rates': per_class_false_rates,
                }

    if best is None:
        raise RuntimeError(
            "No tau/theta pair met the configured maximum per-class known-traffic "
            f"false zero-day rate ({max_allowed:.2f}%). Increase the limit or select another proxy class."
        )

    model.tau_threshold = best['tau']
    model.theta_threshold = best['theta']
    calibration_info = {
        'method': 'proxy_attack_family_holdout_joint_threshold_sweep',
        'proxy_class': class_names[proxy_idx],
        'proxy_validation_samples': int(len(X_pseudo)),
        'proxy_classifier_best_known_val_macro_f1': float(proxy_val_f1),
        'tau': best['tau'],
        'theta': best['theta'],
        'proxy_zero_day_detection_rate_pct': best['pseudo_recall'],
        'max_known_false_zero_day_rate_pct': best['known_false_rate'],
        'max_allowed_known_false_zero_day_rate_pct': max_allowed,
        'known_false_zero_day_rate_by_class_pct': best['per_class_false_rates'],
        'candidate_tau_count': int(len(tau_candidates)),
        'candidate_theta_count': int(len(theta_candidates)),
        'feasible_pair_count': int(feasible_pair_count),
    }
    print(f"  [+] Proxy-calibrated Tau      : {best['tau']:.6f}")
    print(f"  [+] Proxy-calibrated Theta    : {best['theta']:.4f}")
    print(f"  [+] Proxy family detection    : {best['pseudo_recall']:.2f}%")
    print(f"  [+] Max known-class false zero-day rate: {best['known_false_rate']:.2f}% (limit {max_allowed:.2f}%)")
    return calibration_info


def collect_dual_engine_gate_metrics(model, X, normal_idx, device, y=None, batch_size=1024):
    """Measure Engine 1/2 gate rates for known or held-out traffic."""
    model.eval()
    X = torch.as_tensor(X, dtype=torch.float32)
    y = None if y is None else torch.as_tensor(y, dtype=torch.long).cpu().numpy()

    all_verdicts, all_preds, all_confidence, all_recon = [], [], [], []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = X[start:start + batch_size].to(device)
            verdicts, preds, confidence, recon = model.predict_verdict(
                batch, normal_class_idx=normal_idx
            )
            all_verdicts.extend(verdicts.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_confidence.extend(confidence.cpu().numpy().tolist())
            all_recon.extend(recon.cpu().numpy().tolist())

    verdicts = np.asarray(all_verdicts, dtype=np.int64)
    preds = np.asarray(all_preds, dtype=np.int64)
    confidence = np.asarray(all_confidence, dtype=np.float64)
    recon = np.asarray(all_recon, dtype=np.float64)
    engine1_anomaly = recon > model.tau_threshold
    engine2_low_confidence = confidence < model.theta_threshold
    zero_day_flag = verdicts == 2

    def percent(mask):
        return float(np.mean(mask) * 100.0) if len(mask) else 0.0

    after_engine1 = engine1_anomaly
    metrics = {
        "engine1_anomaly_rate_pct": percent(engine1_anomaly),
        "engine2_low_confidence_rate_after_engine1_pct": percent(
            engine2_low_confidence[after_engine1]
        ),
        "engine2_normal_prediction_rate_after_engine1_pct": percent(
            preds[after_engine1] == normal_idx
        ),
        "zero_day_flag_rate_pct": percent(zero_day_flag),
    }

    if y is not None and len(y) == len(verdicts):
        is_benign = y == normal_idx
        is_attack = ~is_benign
        attack_reached_engine2 = is_attack & engine1_anomaly
        metrics.update({
            "known_traffic_false_zero_day_rate_pct": percent(zero_day_flag),
            "benign_false_zero_day_rate_pct": percent(zero_day_flag[is_benign]),
            "known_attack_false_zero_day_rate_pct": percent(zero_day_flag[is_attack]),
            "benign_engine1_anomaly_rate_pct": percent(engine1_anomaly[is_benign]),
            "known_attack_engine1_bypass_rate_pct": percent(~engine1_anomaly[is_attack]),
            "known_attack_engine2_low_confidence_rate_after_engine1_pct": percent(
                engine2_low_confidence[attack_reached_engine2]
            ),
            "known_attack_engine2_normal_prediction_rate_after_engine1_pct": percent(
                preds[attack_reached_engine2] == normal_idx
            ),
        })
    return metrics


def pretrain_engine1(model, X_train, y_train, normal_idx, device, batch_size=64, lr=0.001, epochs=5):
    print("  [*] Pretraining Engine 1 Autoencoder on benign samples...")
    criterion_recon = nn.MSELoss()
    ae_optimizer = optim.Adam(model.engine1_zero_day_guard.parameters(), lr=lr)
    
    benign_mask = (y_train == normal_idx)
    X_train_benign = X_train[benign_mask]
    benign_loader = DataLoader(
        TensorDataset(torch.tensor(X_train_benign, dtype=torch.float32)),
        batch_size=batch_size, shuffle=True, drop_last=len(X_train_benign) >= batch_size
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
            torch.nn.utils.clip_grad_norm_(model.engine1_zero_day_guard.parameters(), 1.0)
            ae_optimizer.step()
            running_loss += loss_ae.item() * x_b.size(0)
        
        if ae_ep % 2 == 0 or ae_ep == epochs:
            avg_loss = running_loss / len(X_train_benign) if len(X_train_benign) > 0 else 0.0
            print(f"      AE Epoch [{ae_ep}/{epochs}] | Benign Recon MSE: {avg_loss:.6f}")


# =====================================================================
# 1. 3-WAY SPLIT PIPELINE (TRAIN / VAL / TEST)
# =====================================================================
def run_single_experiment(args, device):
    set_seed(args.seed)
    balance_suffix = f"{args.balancer}_balanced" if args.balance else "raw"
    ds_prefix = args.dataset_name.lower().replace("-", "_")
    hide_suffix = f"_loco_{args.hide_class.lower()}" if args.hide_class else ""
    run_tag = f"{args.loss}_lr{args.lr:g}_bs{args.batch_size}_ema{args.ema_decay:g}_ar{args.adasyn_ratio:g}_s{args.seed}"
    exp_name = f"{ds_prefix}_{args.model}{hide_suffix}_{balance_suffix}_{run_tag}_{args.epochs}ep"
    results_dir = os.path.join("results", exp_name)
    os.makedirs(results_dir, exist_ok=True)

    print("\n==================================================")
    print(f"    RUNNING EXPERIMENT: {exp_name.upper()}")
    print("==================================================")

    raw_df = clean_target_rows(pd.read_csv(args.dataset), args.target_col)

    zero_day_df = None
    if args.hide_class:
        zero_day_df = raw_df[raw_df[args.target_col].astype(str).str.lower() == args.hide_class.lower()].copy()
        raw_df = raw_df[raw_df[args.target_col].astype(str).str.lower() != args.hide_class.lower()].copy()
        print(f"  [*] Zero-Day Class '{args.hide_class}' isolated: {len(zero_day_df)} samples reserved.")

    class_counts = raw_df[args.target_col].astype(str).str.strip().value_counts()
    if len(class_counts) and class_counts.min() < 7:
        raise ValueError(
            "A stratified train/validation/test split needs at least 7 samples per known class; "
            f"smallest class has {int(class_counts.min())}. Use more data or a two-way split."
        )
    stratify_col = raw_df[args.target_col] if args.target_col in raw_df else None
    train_df, temp_df = train_test_split(raw_df, test_size=0.30, random_state=args.seed, stratify=stratify_col)
    temp_stratify = temp_df[args.target_col] if args.target_col in temp_df else None
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=args.seed, stratify=temp_stratify)

    if args.balance:
        print(f"\n--- STAGE: Balancing via {args.balancer.upper()} (Training Partition Only) ---")
    X_train, X_val, y_train, y_val, X_test, y_test, preprocessor = prepare_training_arrays(
        train_df, val_df, args, args.seed, test_df=test_df
    )

    num_features = X_train.shape[1]
    class_names = [str(c) for c in preprocessor.target_encoder.classes_]
    num_classes = len(class_names)
    
    normal_idx = None
    for idx, c in enumerate(class_names):
        if c.lower() in ['normal', 'benign', 'benigntraffic']:
            normal_idx = idx
            break
    if normal_idx is None:
        raise ValueError(f"Could not identify a benign/normal class in {class_names}.")

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=len(train_dataset) >= args.batch_size)
    train_eval_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)), batch_size=args.batch_size, shuffle=False)

    model = get_model(args.model, input_features=num_features, num_classes=num_classes).to(device)
    criterion_cls = make_classifier_loss(args, y_train, num_classes, device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)

    if args.model.lower() in ['dual-engine', 'dual_engine', 'dual']:
        pretrain_engine1(model, X_train, y_train, normal_idx, device, batch_size=args.batch_size, lr=args.lr, epochs=10)
    ema = ModelEMA(model, decay=args.ema_decay)

    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_macro_f1": [], "val_macro_f1_smoothed": [], "test_loss": [], "test_acc": []}
    best_val_f1 = -1.0
    best_model_state = None
    stale_epochs = 0
    val_f1_window = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model.engine2_classifier(X_b) if hasattr(model, 'engine2_classifier') else model(X_b)
            loss = criterion_cls(logits, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            ema.update(model)

        # Evaluate with dropout disabled and consistent inference behavior for
        # train, validation, and test curve measurements.
        epoch_train_loss, epoch_train_acc, _ = evaluate_classifier(
            ema.module, train_eval_loader, criterion_cls, device
        )

        epoch_val_loss, epoch_val_acc, epoch_val_f1 = evaluate_classifier(
            ema.module, val_loader, criterion_cls, device
        )
        scheduler.step(epoch_val_loss)
        # Test curves are recorded for visibility only; test scores never select
        # checkpoints, tune thresholds, or drive the learning-rate scheduler.
        epoch_test_loss, epoch_test_acc, _ = evaluate_classifier(
            ema.module, test_loader, criterion_cls, device
        )
        val_f1_window.append(epoch_val_f1)
        smoothed_val_f1 = float(np.mean(val_f1_window[-args.smoothing_window:]))

        history["epoch"].append(epoch)
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)
        history["val_macro_f1"].append(epoch_val_f1)
        history["val_macro_f1_smoothed"].append(smoothed_val_f1)
        history["test_loss"].append(epoch_test_loss)
        history["test_acc"].append(epoch_test_acc)

        if smoothed_val_f1 > best_val_f1 + args.min_delta:
            best_val_f1 = smoothed_val_f1
            best_model_state = copy.deepcopy(ema.module.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1

        if epoch % max(1, args.epochs // 10) == 0 or epoch == args.epochs:
            print(f"Epoch [{epoch}/{args.epochs}] | Train Acc: {epoch_train_acc*100:.2f}% | Val Acc: {epoch_val_acc*100:.2f}% | Val Macro-F1: {epoch_val_f1*100:.2f}% (smoothed {smoothed_val_f1*100:.2f}%)")
        if stale_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch}; validation macro-F1 did not improve for {args.patience} epochs.")
            break

    last_model_state = copy.deepcopy(model.state_dict())
    if best_model_state is not None:
        ema.module.load_state_dict(best_model_state)
    model_for_eval = ema.module

    threshold_calibration_info = None
    if args.model.lower() in ['dual-engine', 'dual_engine', 'dual']:
        threshold_calibration_info = calibrate_dual_engine_thresholds_with_proxy(
            model_for_eval, X_train, y_train, X_val, y_val, class_names,
            normal_idx, args, device
        )

    # Evaluate Test Set
    model_for_eval.eval()
    y_preds, y_trues, y_probs = [], [], []
    start_eval = time.time()
    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b = X_b.to(device)
            logits = model_for_eval.engine2_classifier(X_b) if hasattr(model_for_eval, 'engine2_classifier') else model_for_eval(X_b)
            probs = F.softmax(logits, dim=-1)
            _, p = torch.max(probs, 1)
            y_preds.extend(p.cpu().numpy())
            y_trues.extend(y_b.numpy())
            y_probs.extend(probs.cpu().numpy())

    eval_time = time.time() - start_eval
    num_samples = len(y_trues)
    latency_ms = (eval_time / num_samples) * 1000 if num_samples > 0 else 0.0
    throughput = num_samples / eval_time if eval_time > 0 else 0.0

    y_trues = np.array(y_trues)
    y_preds = np.array(y_preds)
    y_probs = np.array(y_probs)

    metrics = compute_metrics(y_trues, y_preds, normal_idx=normal_idx, num_classes=num_classes)
    if threshold_calibration_info is not None:
        metrics['threshold_calibration'] = threshold_calibration_info
    if args.model.lower() in ['dual-engine', 'dual_engine', 'dual']:
        validation_gates = collect_dual_engine_gate_metrics(
            model_for_eval, X_val, normal_idx, device, y=y_val
        )
        test_gates = collect_dual_engine_gate_metrics(
            model_for_eval, X_test, normal_idx, device, y=y_test
        )
        metrics.update({f"validation_{key}": value for key, value in validation_gates.items()})
        metrics.update({f"known_test_{key}": value for key, value in test_gates.items()})

    best_train_loss, best_train_accuracy, _ = evaluate_classifier(
        model_for_eval, train_eval_loader, criterion_cls, device
    )
    metrics['train_accuracy'] = best_train_accuracy
    metrics['train_loss'] = best_train_loss
    metrics['final_train_accuracy'] = best_train_accuracy
    metrics['best_val_macro_f1'] = best_val_f1
    metrics['test_accuracy'] = metrics['accuracy']
    metrics['requested_balancer'] = args.balancer if args.balance else 'none'
    metrics['actual_balancing_methods'] = named_balancing_methods(preprocessor, class_names)
    metrics['inference_latency_ms'] = latency_ms
    metrics['throughput_pps'] = throughput

    # Artifact generation
    with open(os.path.join(results_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=4)
    with open(os.path.join(results_dir, "preprocessor.pkl"), "wb") as f:
        pickle.dump(preprocessor, f)

    plot_learning_curves(history, os.path.join(results_dir, "training_curves.png"))
    plot_confusion_matrix(y_trues, y_preds, class_names, os.path.join(results_dir, "confusion_matrix.png"))
    plot_roc_curves(y_trues, y_probs, class_names, os.path.join(results_dir, "roc_curves.png"))

    rep_df = pd.DataFrame(classification_report(
        y_trues, y_preds, labels=list(range(num_classes)), target_names=class_names,
        output_dict=True, zero_division=0
    )).transpose()
    rep_df.to_csv(os.path.join(results_dir, "classification_report.csv"))

    if zero_day_df is not None and len(zero_day_df) > 0:
        X_zd, _ = preprocessor.transform(zero_day_df)
        zero_day_gates = collect_dual_engine_gate_metrics(
            model_for_eval, X_zd, normal_idx, device
        )
        zero_day_gates['isolation_rate_pct'] = zero_day_gates.pop('zero_day_flag_rate_pct')
        metrics.update({f"zero_day_{key}": value for key, value in zero_day_gates.items()})
        pct = zero_day_gates['isolation_rate_pct']
        metrics['zero_day_isolation_acc'] = pct
        detected = int(round(pct * len(X_zd) / 100.0))
        print(f"  [+] Zero-Day Isolation Rate: {detected}/{len(X_zd)} ({pct:.2f}%)")
        print(
            "  [+] Gate diagnostics: "
            f"Engine 1 passed {zero_day_gates['engine1_anomaly_rate_pct']:.2f}% to Engine 2; "
            f"Engine 2 low confidence on {zero_day_gates['engine2_low_confidence_rate_after_engine1_pct']:.2f}% of those."
        )

    torch.save(best_model_state, os.path.join(results_dir, "best_model.pt"))
    torch.save(last_model_state, os.path.join(results_dir, "last_model.pt"))
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\n[+] Results & Visualizations successfully exported to: {results_dir}\n")


# =====================================================================
# 2. 5-FOLD CV PIPELINE (WITH EXPLICIT TRAIN/TEST STATS & VISUALIZATIONS)
# =====================================================================
def run_kfold_experiment(args, device):
    set_seed(args.seed)
    balance_suffix = f"{args.balancer}_balanced" if args.balance else "raw"
    ds_prefix = args.dataset_name.lower().replace("-", "_")
    hide_suffix = f"_loco_{args.hide_class.lower()}" if args.hide_class else ""
    run_tag = f"{args.loss}_lr{args.lr:g}_bs{args.batch_size}_ema{args.ema_decay:g}_ar{args.adasyn_ratio:g}_s{args.seed}"
    exp_dir_name = f"{ds_prefix}_{args.model}{hide_suffix}_{balance_suffix}_{run_tag}_{args.kfold}fold_{args.epochs}ep"
    results_dir = os.path.join("results", exp_dir_name)
    os.makedirs(results_dir, exist_ok=True)

    print("\n=======================================================")
    print(f"   RUNNING {args.kfold}-FOLD CROSS VALIDATION: {exp_dir_name.upper()}")
    print("=======================================================")

    raw_df = clean_target_rows(pd.read_csv(args.dataset), args.target_col)
    y_raw = raw_df[args.target_col].copy()

    skf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
    fold_metrics = []
    all_fold_histories = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(raw_df, y_raw), 1):
        print(f"\n>>>>>>>>>>>>>>>>>>>> FOLD [{fold}/{args.kfold}] <<<<<<<<<<<<<<<<<<<<")
        fold_dir = os.path.join(results_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        train_df = raw_df.iloc[train_idx].copy()
        val_df = raw_df.iloc[val_idx].copy()

        # Keep the outer fold as untouched test data; the hidden class is held
        # aside for LOCO scoring and is excluded from known-class fitting.
        zero_day_test_df = None
        if args.hide_class:
            zero_day_test_df = val_df[val_df[args.target_col].astype(str).str.lower() == args.hide_class.lower()].copy()
            val_df = val_df[val_df[args.target_col].astype(str).str.lower() != args.hide_class.lower()].copy()
            train_df = train_df[train_df[args.target_col].astype(str).str.lower() != args.hide_class.lower()].copy()

        test_df = val_df
        fold_counts = train_df[args.target_col].astype(str).value_counts()
        if len(fold_counts) and fold_counts.min() < 2:
            raise ValueError("Each training fold needs at least two examples per known class for a stratified validation split.")
        train_df, val_df = train_test_split(
            train_df, test_size=args.validation_fraction, random_state=args.seed + fold,
            stratify=train_df[args.target_col]
        )

        if args.balance:
            print(f"  [*] Applying {args.balancer.upper()} to the inner training partition only...")
        X_train, X_val, y_train, y_val, X_test, y_test, preprocessor = prepare_training_arrays(
            train_df, val_df, args, args.seed + fold, test_df=test_df
        )

        num_features = X_train.shape[1]
        class_names = [str(c) for c in preprocessor.target_encoder.classes_]
        num_classes = len(class_names)
        
        normal_idx = None
        for idx, c in enumerate(class_names):
            if c.lower() in ['normal', 'benign', 'benigntraffic']:
                normal_idx = idx
                break
        if normal_idx is None:
            raise ValueError(f"Could not identify a benign/normal class in {class_names}.")

        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=len(train_dataset) >= args.batch_size)
        train_eval_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
        val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)), batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)), batch_size=args.batch_size, shuffle=False)

        model = get_model(args.model, input_features=num_features, num_classes=num_classes).to(device)
        criterion_cls = make_classifier_loss(args, y_train, num_classes, device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)

        if args.model.lower() in ['dual-engine', 'dual_engine', 'dual']:
            pretrain_engine1(model, X_train, y_train, normal_idx, device, batch_size=args.batch_size, lr=args.lr, epochs=5)
        ema = ModelEMA(model, decay=args.ema_decay)

        history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_macro_f1": [], "val_macro_f1_smoothed": [], "test_loss": [], "test_acc": []}
        best_val_f1 = -1.0
        best_model_state = None
        stale_epochs = 0
        val_f1_window = []

        for epoch in range(1, args.epochs + 1):
            model.train()
            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                logits = model.engine2_classifier(X_b) if hasattr(model, 'engine2_classifier') else model(X_b)
                loss = criterion_cls(logits, y_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                ema.update(model)

            epoch_train_loss, epoch_train_acc, _ = evaluate_classifier(
                ema.module, train_eval_loader, criterion_cls, device
            )

            epoch_val_loss, epoch_val_acc, epoch_val_f1 = evaluate_classifier(
                ema.module, val_loader, criterion_cls, device
            )
            scheduler.step(epoch_val_loss)
            epoch_test_loss, epoch_test_acc, _ = evaluate_classifier(
                ema.module, test_loader, criterion_cls, device
            )
            val_f1_window.append(epoch_val_f1)
            smoothed_val_f1 = float(np.mean(val_f1_window[-args.smoothing_window:]))

            history["epoch"].append(epoch)
            history["train_loss"].append(epoch_train_loss)
            history["train_acc"].append(epoch_train_acc)
            history["val_loss"].append(epoch_val_loss)
            history["val_acc"].append(epoch_val_acc)
            history["val_macro_f1"].append(epoch_val_f1)
            history["val_macro_f1_smoothed"].append(smoothed_val_f1)
            history["test_loss"].append(epoch_test_loss)
            history["test_acc"].append(epoch_test_acc)

            if smoothed_val_f1 > best_val_f1 + args.min_delta:
                best_val_f1 = smoothed_val_f1
                best_model_state = copy.deepcopy(ema.module.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1

            if epoch % max(1, args.epochs // 5) == 0 or epoch == args.epochs:
                print(f"  Epoch [{epoch}/{args.epochs}] | Train Acc: {epoch_train_acc*100:.2f}% | Val Acc: {epoch_val_acc*100:.2f}% | Val Macro-F1: {epoch_val_f1*100:.2f}% (smoothed {smoothed_val_f1*100:.2f}%)")
            if stale_epochs >= args.patience:
                print(f"  Early stopping at epoch {epoch}; validation macro-F1 did not improve for {args.patience} epochs.")
                break

        last_model_state = copy.deepcopy(model.state_dict())
        if best_model_state is not None:
            ema.module.load_state_dict(best_model_state)
        model_for_eval = ema.module

        threshold_calibration_info = None
        if args.model.lower() in ['dual-engine', 'dual_engine', 'dual']:
            threshold_calibration_info = calibrate_dual_engine_thresholds_with_proxy(
                model_for_eval, X_train, y_train, X_val, y_val, class_names,
                normal_idx, args, device
            )

        # Final Fold Inference for Visualizations & Reports
        model_for_eval.eval()
        y_preds, y_trues, y_probs = [], [], []
        with torch.no_grad():
            for X_v, y_v in test_loader:
                X_v = X_v.to(device)
                logits = model_for_eval.engine2_classifier(X_v) if hasattr(model_for_eval, 'engine2_classifier') else model_for_eval(X_v)
                probs = F.softmax(logits, dim=-1)
                _, p = torch.max(probs, 1)
                y_preds.extend(p.cpu().numpy())
                y_trues.extend(y_v.numpy())
                y_probs.extend(probs.cpu().numpy())

        y_trues = np.array(y_trues)
        y_preds = np.array(y_preds)
        y_probs = np.array(y_probs)

        metrics = compute_metrics(y_trues, y_preds, normal_idx=normal_idx, num_classes=num_classes)
        if threshold_calibration_info is not None:
            metrics['threshold_calibration'] = threshold_calibration_info
        metrics['fold'] = fold
        if args.model.lower() in ['dual-engine', 'dual_engine', 'dual']:
            validation_gates = collect_dual_engine_gate_metrics(
                model_for_eval, X_val, normal_idx, device, y=y_val
            )
            known_test_gates = collect_dual_engine_gate_metrics(
                model_for_eval, X_test, normal_idx, device, y=y_test
            )
            metrics.update({f"validation_{key}": value for key, value in validation_gates.items()})
            metrics.update({f"known_test_{key}": value for key, value in known_test_gates.items()})

        train_loss_best, train_accuracy_best, _ = evaluate_classifier(
            model_for_eval, train_eval_loader, criterion_cls, device
        )
        metrics['train_accuracy'] = float(train_accuracy_best)
        metrics['train_loss'] = float(train_loss_best)
        metrics['final_train_accuracy'] = float(train_accuracy_best)
        metrics['test_accuracy'] = float(metrics['accuracy'])
        metrics['requested_balancer'] = args.balancer if args.balance else 'none'
        metrics['actual_balancing_methods'] = named_balancing_methods(preprocessor, class_names)
        metrics['best_val_macro_f1'] = float(best_val_f1)
        metrics['final_test_val_accuracy'] = float(metrics['accuracy'])

        # LOCO Evaluation
        if zero_day_test_df is not None and len(zero_day_test_df) > 0 and args.model.lower() in ['dual-engine', 'dual_engine', 'dual']:
            X_zd, _ = preprocessor.transform(zero_day_test_df)
            zero_day_gates = collect_dual_engine_gate_metrics(
                model_for_eval, X_zd, normal_idx, device
            )
            zero_day_gates['isolation_rate_pct'] = zero_day_gates.pop('zero_day_flag_rate_pct')
            metrics.update({f"zero_day_{key}": value for key, value in zero_day_gates.items()})
            zd_acc = zero_day_gates['isolation_rate_pct']
            metrics['zero_day_isolation_acc'] = float(zd_acc)
            zd_detected = int(round(zd_acc * len(X_zd) / 100.0))
            print(f"  [Fold {fold} Zero-Day LOCO] Detected {zd_detected}/{len(X_zd)} ({zd_acc:.2f}%)")
            print(
                "  [Fold {fold} Gate diagnostics] ".format(fold=fold) +
                f"Engine 1 passed {zero_day_gates['engine1_anomaly_rate_pct']:.2f}% to Engine 2; "
                f"Engine 2 low confidence on {zero_day_gates['engine2_low_confidence_rate_after_engine1_pct']:.2f}% of those."
            )

        fold_metrics.append(metrics)
        all_fold_histories.append(history)

        # Save Fold Models, Data & Plots
        torch.save(best_model_state, os.path.join(fold_dir, "best_model.pt"))
        torch.save(last_model_state, os.path.join(fold_dir, "last_model.pt"))
        with open(os.path.join(fold_dir, "preprocessor.pkl"), "wb") as f:
            pickle.dump(preprocessor, f)
        with open(os.path.join(fold_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=4)

        with open(os.path.join(fold_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)

        plot_learning_curves(history, os.path.join(fold_dir, "training_curves.png"))
        plot_confusion_matrix(y_trues, y_preds, class_names, os.path.join(fold_dir, "confusion_matrix.png"))
        plot_roc_curves(y_trues, y_probs, class_names, os.path.join(fold_dir, "roc_curves.png"))

        rep_df = pd.DataFrame(classification_report(
            y_trues, y_preds, labels=list(range(num_classes)), target_names=class_names,
            output_dict=True, zero_division=0
        )).transpose()
        rep_df.to_csv(os.path.join(fold_dir, "classification_report.csv"))

        print(f"  [Fold {fold} Saved] Train Acc: {metrics['final_train_accuracy']*100:.2f}% | Test Acc: {metrics['accuracy']*100:.2f}% | DR: {metrics['detection_rate']*100:.2f}% | FAR: {metrics['false_alarm_rate']*100:.2f}%")

    train_accs = [m['final_train_accuracy'] * 100 for m in fold_metrics]
    test_accs = [m['accuracy'] * 100 for m in fold_metrics]
    f1s = [m['f1_macro'] * 100 for m in fold_metrics]
    drs = [m['detection_rate'] * 100 for m in fold_metrics]
    fars = [m['false_alarm_rate'] * 100 for m in fold_metrics]

    summary = {
        "mean_train_accuracy": float(np.mean(train_accs)),
        "std_train_accuracy": float(np.std(train_accs)),
        "mean_test_accuracy": float(np.mean(test_accs)),
        "std_test_accuracy": float(np.std(test_accs)),
        "mean_f1_macro": float(np.mean(f1s)),
        "std_f1_macro": float(np.std(f1s)),
        "mean_detection_rate": float(np.mean(drs)),
        "std_detection_rate": float(np.std(drs)),
        "mean_false_alarm_rate": float(np.mean(fars)),
        "std_false_alarm_rate": float(np.std(fars)),
        "folds": fold_metrics
    }

    # Aggregate gate diagnostics across folds while retaining the per-fold
    # values above for identifying which stage limits zero-day detection.
    diagnostic_prefixes = ('validation_', 'known_test_', 'zero_day_')
    diagnostic_keys = [
        key for key, value in fold_metrics[0].items()
        if key.startswith(diagnostic_prefixes) and isinstance(value, (int, float))
    ] if fold_metrics else []
    for key in diagnostic_keys:
        values = [float(m[key]) for m in fold_metrics if key in m]
        if values:
            summary[f"mean_{key}"] = float(np.mean(values))
            summary[f"std_{key}"] = float(np.std(values))

    if args.hide_class and 'zero_day_isolation_acc' in fold_metrics[0]:
        zd_accs = [m['zero_day_isolation_acc'] for m in fold_metrics]
        summary["mean_zero_day_isolation_acc"] = float(np.mean(zd_accs))
        summary["std_zero_day_isolation_acc"] = float(np.std(zd_accs))

    with open(os.path.join(results_dir, "kfold_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=4)

    print("\n=======================================================")
    print(f"        {args.kfold}-FOLD CROSS VALIDATION SUMMARY       ")
    print("=======================================================")
    print(f"  TRAIN ACCURACY : {summary['mean_train_accuracy']:.2f}% (+/- {summary['std_train_accuracy']:.2f}%)")
    print(f"  TEST ACCURACY  : {summary['mean_test_accuracy']:.2f}% (+/- {summary['std_test_accuracy']:.2f}%)")
    print(f"  MACRO F1       : {summary['mean_f1_macro']:.2f}% (+/- {summary['std_f1_macro']:.2f}%)")
    print(f"  DETECTION RT   : {summary['mean_detection_rate']:.2f}% (+/- {summary['std_detection_rate']:.2f}%)")
    print(f"  FALSE ALARM    : {summary['mean_false_alarm_rate']:.2f}% (+/- {summary['std_false_alarm_rate']:.2f}%)")
    if "mean_zero_day_isolation_acc" in summary:
        print(f"  ZERO-DAY ACC   : {summary['mean_zero_day_isolation_acc']:.2f}% (+/- {summary['std_zero_day_isolation_acc']:.2f}%)")
    print(f"  Artifacts saved to: {results_dir}")
    print("=======================================================\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dual-Engine Network Intrusion Detection System")
    parser.add_argument('--dataset', type=str, default='data/UNSW_NB15_combined.csv')
    parser.add_argument('--dataset_name', type=str, default='UNSW-NB15', choices=['UNSW-NB15', 'CIC-IDS2018', 'CIC-IOT2023'])
    parser.add_argument('--model', type=str, default='dual-engine', choices=['proposed', 'dual-engine', 'cnn', 'resnet', 'resnest', 'resnet-gru'])
    parser.add_argument('--target_col', type=str, default='attack_cat')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--patience', type=int, default=12, help='Early stopping patience on validation macro-F1')
    parser.add_argument('--min_delta', type=float, default=0.001, help='Minimum macro-F1 gain to reset early stopping')
    parser.add_argument('--smoothing_window', type=int, default=3, help='Trailing validation macro-F1 averaging window')
    parser.add_argument('--validation_fraction', type=float, default=0.15, help='Inner validation fraction for each outer CV training fold')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Maximum gradient norm')
    parser.add_argument('--ema_decay', type=float, default=0.99, help='EMA decay used for validation and best checkpoint')
    parser.add_argument('--adasyn_ratio', type=float, default=0.1, help='Minority oversampling cap relative to majority count (default: 10%)')
    parser.add_argument('--loss', choices=['cross_entropy', 'weighted', 'focal'], default='weighted')
    parser.add_argument('--label_smoothing', type=float, default=0.05)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--balance', action='store_true', help='Enable minority class balancing')
    parser.add_argument('--balancer', type=str, default='adasyn', choices=['adasyn', 'smotenc', 'random', 'auto', 'ctgan'])
    parser.add_argument('--ctgan_epochs', type=int, default=10)
    parser.add_argument('--hide_class', type=str, default=None, help='Class to hide for zero-day LOCO test')
    parser.add_argument('--calibration_class', type=str, default='auto', help='Known attack family held out from the proxy calibration classifier; auto selects the largest validation attack family')
    parser.add_argument('--calibration_max_false_zero_day_rate', type=float, default=5.0, help='Maximum per-class known-validation false zero-day rate allowed during joint threshold calibration (percent)')
    parser.add_argument('--kfold', type=int, default=0, help='Set > 1 (e.g., 5) for K-Fold CV; 0 for 3-way split')
    args = parser.parse_args()

    if args.batch_size < 2 or args.patience < 1 or args.smoothing_window < 1:
        parser.error('batch_size must be >= 2 and patience/smoothing_window must be positive.')
    if not 0.0 < args.validation_fraction < 0.5:
        parser.error('validation_fraction must be between 0 and 0.5.')
    if not 0.0 < args.adasyn_ratio <= 1.0:
        parser.error('adasyn_ratio must be in (0, 1].')
    if not 0.0 <= args.calibration_max_false_zero_day_rate <= 100.0:
        parser.error('calibration_max_false_zero_day_rate must be between 0 and 100.')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device}")

    if args.kfold > 1:
        run_kfold_experiment(args, device)
    else:
        run_single_experiment(args, device)
