import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def compute_metrics(y_true, y_pred, normal_idx=0, num_classes=None, class_labels=None):
    """
    Robust Metric Evaluation for NIDS:
    - Sets confusion matrix dimensions across full class space using `labels` to prevent index mismatch.
    - Handles missing classes in zero-day LOCO or imbalanced subsets without crashing.
    - Accurately decouples False Alarm Rate (FAR on benign) and Detection Rate (DR on attacks).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if class_labels is None:
        if num_classes is not None:
            class_labels = list(range(num_classes))
        else:
            all_classes = np.union1d(np.unique(y_true), np.unique(y_pred))
            class_labels = list(range(int(all_classes.max()) + 1)) if len(all_classes) > 0 else [0]

    # Compute CM with fixed class labels
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)

    acc = accuracy_score(y_true, y_pred) if len(y_true) > 0 else 0.0
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=class_labels, average='macro', zero_division=0
    )

    # 1. False Alarm Rate (FAR): False Positives on Normal / Total True Normal
    if normal_idx in class_labels:
        n_idx = class_labels.index(normal_idx)
        total_benign = cm[n_idx, :].sum()
        fp_benign = total_benign - cm[n_idx, n_idx]
        far = float(fp_benign / total_benign) if total_benign > 0 else 0.0
    else:
        far = 0.0

    # 2. Detection Rate (DR): Correctly identified attack traffic / Total actual attack traffic
    attack_mask = (y_true != normal_idx)
    total_attacks = int(np.sum(attack_mask))

    if total_attacks > 0:
        attack_correct = int(np.sum((y_pred[attack_mask] != normal_idx) & (y_pred[attack_mask] == y_true[attack_mask])))
        dr = float(attack_correct / total_attacks)
    else:
        dr = 1.0 if len(y_true) == 0 else float(rec)

    return {
        'accuracy': float(acc),
        'precision_macro': float(prec),
        'recall_macro': float(rec),
        'f1_macro': float(f1),
        'detection_rate': float(dr),
        'false_alarm_rate': float(far),
        'confusion_matrix': cm.tolist()
    }

def save_experiment_results(exp_name, metrics, y_true, y_pred, train_accs, train_losses,
                            class_names, val_accs=None, val_losses=None,
                            best_model_state=None, last_model_state=None):
    save_dir = os.path.join("results", exp_name)
    os.makedirs(save_dir, exist_ok=True)

    # Save PyTorch checkpoints
    if best_model_state is not None:
        best_path = os.path.join(save_dir, "best_model.pt")
        torch.save(best_model_state, best_path)
        print(f"  [+] Saved Best Model checkpoint: {best_path}")

    if last_model_state is not None:
        last_path = os.path.join(save_dir, "last_model.pt")
        torch.save(last_model_state, last_path)
        print(f"  [+] Saved Last Model checkpoint: {last_path}")

    # Save Epoch Progression CSV
    epochs_range = list(range(1, len(train_accs) + 1))
    history_df = pd.DataFrame({
        "epoch": epochs_range,
        "train_loss": train_losses,
        "train_acc": train_accs,
        "val_loss": val_losses if val_losses is not None else [0.0] * len(epochs_range),
        "val_acc": val_accs if val_accs is not None else [0.0] * len(epochs_range)
    })
    history_path = os.path.join(save_dir, "epoch_history.csv")
    history_df.to_csv(history_path, index=False)

    # Plot and Save Dual Accuracy & Loss Curves
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label="Train Loss", color="royalblue", lw=2)
    if val_losses:
        plt.plot(epochs_range, val_losses, label="Val Loss", color="crimson", linestyle="--", lw=2)
    plt.title(f"Loss - {exp_name}", fontsize=11, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, [a * 100 for a in train_accs], label="Train Acc (%)", color="forestgreen", lw=2)
    if val_accs:
        plt.plot(epochs_range, [a * 100 for a in val_accs], label="Val Acc (%)", color="darkorange", linestyle="--", lw=2)
    plt.title(f"Accuracy - {exp_name}", fontsize=11, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()

    plt.tight_layout()
    curve_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(curve_path, dpi=300)
    plt.close()

    # Save metrics JSON
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # Master Experiment Summary CSV
    master_summary = os.path.join("results", "experiment_summary.csv")
    summary_entry = {
        "experiment": exp_name,
        "test_accuracy": metrics.get('test_accuracy', 0.0),
        "f1_macro": metrics.get('f1_macro', 0.0),
        "dr": metrics.get('detection_rate', 0.0),
        "far": metrics.get('false_alarm_rate', 0.0),
        "inference_ms": metrics.get('inference_latency_ms', 0.0),
        "throughput_pps": metrics.get('throughput_pps', 0.0)
    }
    master_df = pd.DataFrame([summary_entry])
    if os.path.exists(master_summary):
        master_df.to_csv(master_summary, mode='a', header=False, index=False)
    else:
        master_df.to_csv(master_summary, mode='w', header=True, index=False)

def print_paper_comparison_table(dataset_name, model_name, epoch_str, metrics):
    print(f"\n| Dataset: {dataset_name} | Model: {model_name} ({epoch_str}) |")
    print(f"| Accuracy: {metrics['accuracy']*100:.2f}% | F1: {metrics['f1_macro']*100:.2f}% | DR: {metrics['detection_rate']*100:.2f}% | FAR: {metrics['false_alarm_rate']*100:.2f}% |")