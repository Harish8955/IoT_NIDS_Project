import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def compute_metrics(y_true, y_pred, normal_idx=0):
    """
    Computes standard evaluation metrics:
    - Overall Test Accuracy
    - Macro Precision, Recall, and F1-Score
    - Detection Rate (DR): Recall strictly across all attack samples (y_true != normal_idx)
    - False Alarm Rate (FAR): FP / (FP + TN) strictly on benign samples (y_true == normal_idx)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    num_classes = cm.shape[0]

    # Dynamically compute False Alarm Rate (FAR) for benign class
    if 0 <= normal_idx < num_classes:
        fp = cm[:, normal_idx].sum() - cm[normal_idx, normal_idx]
        tn = cm.sum() - (cm[normal_idx, :].sum() + cm[:, normal_idx].sum() - cm[normal_idx, normal_idx])
        # Direct benign FP / (FP + TN)
        fp_benign = cm[normal_idx, :].sum() - cm[normal_idx, normal_idx]
        tn_benign = cm[normal_idx, normal_idx]
        far = fp_benign / (fp_benign + tn_benign) if (fp_benign + tn_benign) > 0 else 0.0
    else:
        far = 0.0

    # Detection Rate: Correctly identified attack traffic / total actual attack traffic
    attack_mask = (y_true != normal_idx)
    if np.sum(attack_mask) > 0:
        attack_correct = np.sum((y_pred[attack_mask] != normal_idx) & (y_pred[attack_mask] == y_true[attack_mask]))
        detection_rate = float(attack_correct / np.sum(attack_mask))
    else:
        detection_rate = float(rec)

    return {
        'accuracy': float(acc),
        'precision_macro': float(prec),
        'recall_macro': float(rec),
        'f1_macro': float(f1),
        'detection_rate': float(detection_rate),
        'false_alarm_rate': float(far)
    }

def save_experiment_results(exp_name, metrics, y_true, y_pred, train_accs, train_losses,
                            class_names, val_accs=None, val_losses=None,
                            best_model_state=None, last_model_state=None):
    save_dir = os.path.join("results", exp_name)
    os.makedirs(save_dir, exist_ok=True)

    if best_model_state is not None:
        torch.save(best_model_state, os.path.join(save_dir, "best_model.pt"))
        print(f"  [+] Saved Best Model checkpoint: {os.path.join(save_dir, 'best_model.pt')}")

    if last_model_state is not None:
        torch.save(last_model_state, os.path.join(save_dir, "last_model.pt"))
        print(f"  [+] Saved Last Model checkpoint: {os.path.join(save_dir, 'last_model.pt')}")

    epochs_range = list(range(1, len(train_accs) + 1))
    history_df = pd.DataFrame({
        "epoch": epochs_range,
        "train_loss": train_losses,
        "train_acc": train_accs,
        "val_loss": val_losses if val_losses is not None else [0.0] * len(epochs_range),
        "val_acc": val_accs if val_accs is not None else [0.0] * len(epochs_range)
    })
    history_df.to_csv(os.path.join(save_dir, "epoch_history.csv"), index=False)

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label="Train Loss", color="royalblue", lw=2)
    if val_losses:
        plt.plot(epochs_range, val_losses, label="Val Loss", color="crimson", linestyle="--", lw=2)
    plt.title(f"Loss - {exp_name}", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, [a * 100 for a in train_accs], label="Train Acc (%)", color="forestgreen", lw=2)
    if val_accs:
        plt.plot(epochs_range, [a * 100 for a in val_accs], label="Val Acc (%)", color="darkorange", linestyle="--", lw=2)
    plt.title(f"Accuracy - {exp_name}", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=300)
    plt.close()

    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

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