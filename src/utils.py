import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    # Assuming Normal class is index 0
    fp = cm[0, 1:].sum() if cm.shape[0] > 1 else 0
    tn = cm[0, 0] if cm.shape[0] > 0 else 1
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    dr = rec  # Recall across attack categories
    
    return {
        'accuracy': float(acc),
        'precision_macro': float(prec),
        'recall_macro': float(rec),
        'f1_macro': float(f1),
        'detection_rate': float(dr),
        'false_alarm_rate': float(far)
    }

def save_experiment_results(exp_name, metrics, y_true, y_pred, train_accs, train_losses,
                            class_names, test_accs=None, test_losses=None,
                            best_model_state=None, last_model_state=None):
    """
    Saves:
    1. best_model.pt
    2. last_model.pt
    3. epoch_history.csv (train acc, test acc, train loss, test loss per epoch)
    4. training_curves.png (dual-panel plot of Loss and Accuracy curves)
    5. metrics.json & confusion_matrix.png
    """
    save_dir = os.path.join("results", exp_name)
    os.makedirs(save_dir, exist_ok=True)

    # 1. Save PyTorch Model Weights (.pt)
    if best_model_state is not None:
        best_path = os.path.join(save_dir, "best_model.pt")
        torch.save(best_model_state, best_path)
        print(f"  [+] Saved Best Model checkpoint: {best_path}")

    if last_model_state is not None:
        last_path = os.path.join(save_dir, "last_model.pt")
        torch.save(last_model_state, last_path)
        print(f"  [+] Saved Last Model checkpoint: {last_path}")

    # 2. Save Epoch History CSV
    epochs_range = list(range(1, len(train_accs) + 1))
    history_df = pd.DataFrame({
        "epoch": epochs_range,
        "train_loss": train_losses,
        "train_acc": train_accs,
        "test_loss": test_losses if test_losses is not None else [0.0] * len(epochs_range),
        "test_acc": test_accs if test_accs is not None else [0.0] * len(epochs_range)
    })
    history_csv = os.path.join(save_dir, "epoch_history.csv")
    history_df.to_csv(history_csv, index=False)
    print(f"  [+] Saved Training & Testing Accuracy/Loss logs to: {history_csv}")

    # 3. Plot & Save Training Curves
    plt.figure(figsize=(14, 5))

    # Loss Curve Subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label="Train Loss", color="royalblue", lw=2)
    if test_losses:
        plt.plot(epochs_range, test_losses, label="Test Loss", color="crimson", linestyle="--", lw=2)
    plt.title(f"Loss Progression - {exp_name}", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()

    # Accuracy Curve Subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, [a * 100 for a in train_accs], label="Train Accuracy (%)", color="forestgreen", lw=2)
    if test_accs:
        plt.plot(epochs_range, [a * 100 for a in test_accs], label="Test Accuracy (%)", color="darkorange", linestyle="--", lw=2)
    plt.title(f"Accuracy Progression - {exp_name}", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"  [+] Saved Loss & Accuracy Curves to: {plot_path}")

    # 4. Save Final Summary Metrics JSON
    metrics_path = os.path.join(save_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # 5. Append Row to Master Experiment Summary CSV
    master_summary = os.path.join("results", "experiment_summary.csv")
    summary_entry = {
        "experiment": exp_name,
        "best_test_acc": max(test_accs) if test_accs else metrics.get('test_accuracy', 0.0),
        "final_test_acc": metrics.get('test_accuracy', 0.0),
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