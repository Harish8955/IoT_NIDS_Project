import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

    return {
        'accuracy': acc,
        'precision_macro': prec_macro,
        'recall_macro': rec_macro,
        'f1_macro': f1_macro,
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

def save_experiment_results(exp_name, metrics, y_true, y_pred, train_accs, train_losses, class_names):
    """
    Automatically creates a dedicated folder for each experiment run and appends
    metrics to a master CSV table ('results/experiment_summary.csv').
    """
    exp_dir = os.path.join("results", exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # 1. Save Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix — {exp_name}")
    plt.xlabel("Predicted Label (P)")
    plt.ylabel("True Label (T)")
    plt.tight_layout()
    cm_path = os.path.join(exp_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # 2. Save Training Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    epochs = range(1, len(train_accs) + 1)
    ax1.plot(epochs, train_accs, 'b-', label='Accuracy')
    ax1.set_title(f"Accuracy — {exp_name}")
    ax1.grid(True)
    ax2.plot(epochs, train_losses, 'r-', label='Loss')
    ax2.set_title(f"Loss — {exp_name}")
    ax2.grid(True)
    plt.tight_layout()
    curves_path = os.path.join(exp_dir, "training_curves.png")
    plt.savefig(curves_path, dpi=300)
    plt.close()

    # 3. Save JSON metrics
    metrics_path = os.path.join(exp_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    # 4. Append to Master Summary CSV
    row_data = {
        'Experiment': exp_name,
        'Train Accuracy': round(metrics.get('train_accuracy', 0.0) * 100, 2),
        'Test Accuracy': round(metrics['accuracy'] * 100, 2),
        'Precision (Macro)': round(metrics['precision_macro'] * 100, 2),
        'Recall (Macro)': round(metrics['recall_macro'] * 100, 2),
        'F1-Score (Macro)': round(metrics['f1_macro'] * 100, 2),
        'Precision (Weighted)': round(metrics['precision_weighted'] * 100, 2),
        'Recall (Weighted)': round(metrics['recall_weighted'] * 100, 2),
        'F1-Score (Weighted)': round(metrics['f1_weighted'] * 100, 2)
    }
    summary_csv = os.path.join("results", "experiment_summary.csv")
    if os.path.exists(summary_csv):
        summary_df = pd.read_csv(summary_csv)
        # Overwrite row if experiment already ran, else append
        summary_df = summary_df[summary_df['Experiment'] != exp_name]
        summary_df = pd.concat([summary_df, pd.DataFrame([row_data])], ignore_index=True)
    else:
        summary_df = pd.DataFrame([row_data])

    summary_df.to_csv(summary_csv, index=False)
    print(f"Results for '{exp_name}' successfully organized & saved to: {exp_dir}")
    print(f"Master summary updated at: {summary_csv}")
