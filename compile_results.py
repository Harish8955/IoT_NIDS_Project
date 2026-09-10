import os
import json
import pandas as pd
from src.utils import compile_all_experiment_summaries

def main():
    print("==========================================================================")
    print("   COMPILING & AUDITING EXPERIMENTAL RESULTS (TRAIN vs TEST ACCURACY)")
    print("==========================================================================")
    
    results_dir = "results"
    if not os.path.exists(results_dir):
        print("No results directory found.")
        return

    folders = [f for f in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, f))]
    print(f"Total Experiment Folders Found: {len(folders)}")

    updated_count = 0
    for folder in sorted(folders):
        metrics_path = os.path.join(results_dir, folder, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                data = json.load(f)
            
            modified = False
            if 'test_accuracy' not in data:
                data['test_accuracy'] = data.get('accuracy', 0.0)
                modified = True
            
            if modified:
                with open(metrics_path, 'w') as f:
                    json.dump(data, f, indent=4)
                updated_count += 1

    print(f"Updated {updated_count} metrics.json files to include explicit 'test_accuracy' key.")

    # Re-compile master summary CSV
    df = compile_all_experiment_summaries()
    
    if df is not None and not df.empty:
        print("\n--- MASTER EXPERIMENT SUMMARY TABLE (PREVIEW) ---")
        cols_to_show = ['Experiment', 'Train Accuracy', 'Test Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)', 'False Alarm Rate (%)', 'Detection Rate (%)']
        cols = [c for c in cols_to_show if c in df.columns]
        print(df[cols].to_string(index=False))
        print(f"\nTotal runs in summary: {len(df)}")
        print(f"Master CSV saved at: {os.path.join(results_dir, 'experiment_summary.csv')}")

if __name__ == '__main__':
    main()
