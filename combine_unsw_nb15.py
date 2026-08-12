import os
import pandas as pd

def combine_unsw_files(data_dir="data", output_path="data/UNSW_NB15_combined.csv"):
    """
    Merges UNSW_NB15_training-set.csv and UNSW_NB15_testing-set.csv into a single master file.
    """
    train_file = os.path.join(data_dir, "UNSW_NB15_training-set.csv")
    test_file = os.path.join(data_dir, "UNSW_NB15_testing-set.csv")

    dfs = []
    if os.path.exists(train_file):
        print(f"Reading {train_file}...")
        dfs.append(pd.read_csv(train_file))
    
    if os.path.exists(test_file):
        print(f"Reading {test_file}...")
        dfs.append(pd.read_csv(test_file))

    if not dfs:
        print(f"[ERROR] Neither training nor testing UNSW CSV found in '{data_dir}'.")
        return

    combined_df = pd.concat(dfs, ignore_index=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined_df.to_csv(output_path, index=False)
    
    print(f"\n==================================================")
    print(f"SUCCESSFULLY COMBINED UNSW-NB15 MASTER DATASET!")
    print(f"Saved To: {output_path}")
    print(f"Total Master Dataset Rows: {len(combined_df)}")
    if 'attack_cat' in combined_df.columns:
        print("\nCombined Class Distribution ('attack_cat'):")
        print(combined_df['attack_cat'].value_counts())
    print(f"==================================================")

if __name__ == '__main__':
    combine_unsw_files()
