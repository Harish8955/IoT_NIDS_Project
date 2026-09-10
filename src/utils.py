import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, matthews_corrcoef

from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# =====================================================================
# EXACT PAPER PUBLISHED BENCHMARK RESULTS (WANG ET AL., 2024)
# =====================================================================

PAPER_PUBLISHED_RESULTS = {
    'UNSW-NB15': {
        '200epoch': {
            'CNN':       {'accuracy': 87.60, 'precision': 76.21, 'recall': 69.78, 'f1': 70.93},
            'ResNet':    {'accuracy': 88.85, 'precision': 77.41, 'recall': 70.18, 'f1': 71.80},
            'ResNeSt':   {'accuracy': 90.69, 'precision': 79.88, 'recall': 77.47, 'f1': 78.30},
            'ResNet-GRU':{'accuracy': 88.58, 'precision': 77.73, 'recall': 68.22, 'f1': 70.57},
            'Proposed':  {'accuracy': 91.08, 'precision': 79.17, 'recall': 81.93, 'f1': 80.40}
        }
    },
    'CIC-IDS2018': {
        '200epoch': {
            'CNN':       {'accuracy': 92.85, 'precision': 87.73, 'recall': 81.39, 'f1': 81.20},
            'ResNet':    {'accuracy': 91.50, 'precision': 82.00, 'recall': 77.74, 'f1': 73.38},
            'ResNeSt':   {'accuracy': 90.80, 'precision': 82.79, 'recall': 76.25, 'f1': 75.02},
            'ResNet-GRU':{'accuracy': 89.24, 'precision': 78.47, 'recall': 77.38, 'f1': 74.42},
            'Proposed':  {'accuracy': 94.55, 'precision': 86.83, 'recall': 83.58, 'f1': 81.12}
        }
    },
    'CIC-IOT2023': {
        '100epoch': {
            'CNN':       {'accuracy': 95.20, 'precision': 86.23, 'recall': 87.69, 'f1': 85.77},
            'ResNet':    {'accuracy': 95.80, 'precision': 86.10, 'recall': 86.79, 'f1': 84.17},
            'ResNeSt':   {'accuracy': 97.14, 'precision': 89.02, 'recall': 87.58, 'f1': 86.89},
            'ResNet-GRU':{'accuracy': 96.91, 'precision': 88.13, 'recall': 88.82, 'f1': 87.17},
            'Proposed':  {'accuracy': 97.47, 'precision': 89.06, 'recall': 88.28, 'f1': 87.58}
        }
    }
}

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    prec_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    mcc = matthews_corrcoef(y_true, y_pred)

    # Compute False Alarm Rate (FAR / FPR) from confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fp = cm.sum(axis=0) - np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)
    tp = np.diag(cm)
    tn = cm.sum() - (fp + fn + tp)
    
    fpr_per_class = fp / (fp + tn + 1e-10)
    far_macro = float(np.mean(fpr_per_class))

    return {
        'accuracy': acc,
        'test_accuracy': acc,
        'precision_macro': prec_macro,
        'recall_macro': rec_macro,
        'f1_macro': f1_macro,
        'detection_rate': rec_macro,
        'false_alarm_rate': far_macro,
        'mcc': mcc,
        'precision_weighted': prec_weighted,
        'recall_weighted': rec_weighted,
        'f1_weighted': f1_weighted
    }

def print_paper_comparison_table(dataset_name, model_name, epoch_key, experimental_metrics):
    print("\n==========================================================================")
    print(f"   COMPARISON: EXPERIMENTAL RUN vs PAPER PUBLISHED BENCHMARK ({dataset_name})")
    print("==========================================================================")
    print(f"{'Metric':<20} | {'Your Experimental Run':<22} | {'Paper Published Target':<22}")
    print("-" * 72)

    exp_acc = f"{experimental_metrics['accuracy'] * 100:.2f}%"
    exp_prec = f"{experimental_metrics['precision_macro'] * 100:.2f}%"
    exp_rec = f"{experimental_metrics['recall_macro'] * 100:.2f}%"
    exp_f1 = f"{experimental_metrics['f1_macro'] * 100:.2f}%"

    paper_data = PAPER_PUBLISHED_RESULTS.get(dataset_name, {}).get(epoch_key, {}).get(model_name, None)
    
    if paper_data:
        p_acc = f"{paper_data['accuracy']:.2f}%"
        p_prec = f"{paper_data['precision']:.2f}%"
        p_rec = f"{paper_data['recall']:.2f}%"
        p_f1 = f"{paper_data['f1']:.2f}%"
    else:
        p_acc, p_prec, p_rec, p_f1 = "N/A", "N/A", "N/A", "N/A"

    print(f"{'Accuracy (ACC)':<20} | {exp_acc:<22} | {p_acc:<22}")
    print(f"{'Macro Precision':<20} | {exp_prec:<22} | {p_prec:<22}")
    print(f"{'Macro Recall':<20} | {exp_rec:<22} | {p_rec:<22}")
    print(f"{'Macro F1-Score':<20} | {exp_f1:<22} | {p_f1:<22}")
    print("==========================================================================\n")

def compile_all_experiment_summaries():
    """
    Scans all experiment directories in 'results/' and generates/updates
    the master CSV table ('results/experiment_summary.csv') ensuring all
    train and test metrics are included for all runs.
    """
    results_dir = "results"
    if not os.path.exists(results_dir):
        print(f"No results directory found at '{results_dir}'.")
        return None

    folders = [f for f in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, f))]
    rows = []

    for exp_name in sorted(folders):
        metrics_path = os.path.join(results_dir, exp_name, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            test_acc = metrics.get('test_accuracy', metrics.get('accuracy', 0.0))
            
            row = {
                'Experiment': exp_name,
                'Train Accuracy': round(metrics.get('train_accuracy', 0.0) * 100, 2),
                'Test Accuracy': round(test_acc * 100, 2),
                'Precision (Macro)': round(metrics.get('precision_macro', 0.0) * 100, 2),
                'Recall (Macro)': round(metrics.get('recall_macro', 0.0) * 100, 2),
                'F1-Score (Macro)': round(metrics.get('f1_macro', 0.0) * 100, 2),
                'False Alarm Rate (%)': round(metrics.get('false_alarm_rate', 0.0) * 100, 2),
                'Detection Rate (%)': round(metrics.get('detection_rate', 0.0) * 100, 2),
                'MCC': round(metrics.get('mcc', 0.0), 4),
                'Inference Latency (ms)': round(metrics.get('inference_latency_ms', 0.0), 4),
                'Throughput (Packets/Sec)': round(metrics.get('throughput_pps', 0.0), 2),
                'Precision (Weighted)': round(metrics.get('precision_weighted', 0.0) * 100, 2),
                'Recall (Weighted)': round(metrics.get('recall_weighted', 0.0) * 100, 2),
                'F1-Score (Weighted)': round(metrics.get('f1_weighted', 0.0) * 100, 2)
            }
            rows.append(row)

    if rows:
        summary_df = pd.DataFrame(rows)
        summary_csv = os.path.join(results_dir, "experiment_summary.csv")
        summary_df.to_csv(summary_csv, index=False)
        print(f"Successfully compiled {len(rows)} experiment runs into master summary: {summary_csv}")
        return summary_df
    return None

def save_experiment_results(exp_name, metrics, y_true, y_pred, train_accs, train_losses, class_names, test_accs=None, test_losses=None):
    """
    Automatically creates a dedicated folder for each experiment run and appends
    metrics to a master CSV table ('results/experiment_summary.csv').
    Plots high-resolution learning curves comparing Train vs Test Accuracy & Loss per epoch.
    """
    exp_dir = os.path.join("results", exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # 1. Save Confusion Matrix (Raw Counts & Normalized)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix (Raw Counts) — {exp_name}")
    plt.xlabel("Predicted Label (P)")
    plt.ylabel("True Label (T)")
    plt.tight_layout()
    cm_path = os.path.join(exp_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # 1b. Save Normalized Confusion Matrix (Percentages)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Normalized Confusion Matrix (%) — {exp_name}")
    plt.xlabel("Predicted Label (P)")
    plt.ylabel("True Label (T)")
    plt.tight_layout()
    cm_norm_path = os.path.join(exp_dir, "confusion_matrix_normalized.png")
    plt.savefig(cm_norm_path, dpi=300)
    plt.close()

    # 2. Save Training & Testing Curves (Train vs Test Accuracy & Loss)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    epochs = range(1, len(train_accs) + 1)
    
    # Accuracy Curve Subplot
    ax1.plot(epochs, [a * 100 if a <= 1.0 else a for a in train_accs], 'b-o', markersize=3, label='Train Accuracy')
    if test_accs and len(test_accs) == len(train_accs):
        ax1.plot(epochs, [a * 100 if a <= 1.0 else a for a in test_accs], 'g--s', markersize=3, label='Test Accuracy')
    ax1.set_title(f"Accuracy Curve — {exp_name}", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy (%)")
    ax1.legend(loc='best')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Loss Curve Subplot
    ax2.plot(epochs, train_losses, 'r-o', markersize=3, label='Train Loss')
    if test_losses and len(test_losses) == len(train_losses):
        ax2.plot(epochs, test_losses, color='orange', linestyle='--', marker='s', markersize=3, label='Test Loss')
    ax2.set_title(f"Loss Curve — {exp_name}", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend(loc='best')
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    curves_path = os.path.join(exp_dir, "training_curves.png")
    plt.savefig(curves_path, dpi=300)
    plt.close()

    # Ensure test_accuracy key is present in metrics dictionary
    if 'test_accuracy' not in metrics:
        metrics['test_accuracy'] = metrics.get('accuracy', 0.0)

    if test_accs:
        metrics['epoch_test_accuracies'] = test_accs
    if test_losses:
        metrics['epoch_test_losses'] = test_losses
    metrics['epoch_train_accuracies'] = train_accs
    metrics['epoch_train_losses'] = train_losses

    # 3. Save JSON metrics
    metrics_path = os.path.join(exp_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    # 4. Re-compile Master Summary CSV
    compile_all_experiment_summaries()
    print(f"Results & Test Curves for '{exp_name}' successfully saved to: {exp_dir}")

