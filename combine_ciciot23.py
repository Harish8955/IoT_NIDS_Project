import os
import glob
import pandas as pd

def combine_ciciot23_files(base_dir="data/CICIOT23", output_path="data/CICIOT23_combined.csv", max_majority_class=10000):
    """
    Streamed, memory-efficient merger for CICIOT23 train/test CSV files.
    - Preserves 100% of minority attack classes.
    - Samples large majority classes to a manageable limit.
    - Saves a single lightweight master file CICIOT23_combined.csv.
    """
    csv_files = glob.glob(os.path.join(base_dir, "**", "*.csv"), recursive=True)
    if not csv_files:
        print(f"\n[ERROR] No CSV files found in '{base_dir}'.")
        return

    print(f"\nFound {len(csv_files)} CICIOT23 CSV files in '{base_dir}'. Stream-merging...")
    sampled_chunks = []

    ALWAYS_KEEP_CLASSES = {
        'DDoS-HTTP_Flood', 'DDoS-SlowLoris', 'DoS-HTTP_Flood', 
        'Recon-HostDiscovery', 'Recon-PingSweep', 'VulnerabilityScan',
        'DDoS-HTTP Flood', 'DDoS-SlowLoris', 'DoS-HTTP Flood'
    }

    for idx, filepath in enumerate(csv_files, 1):
        filename = os.path.basename(filepath)
        print(f"[{idx}/{len(csv_files)}] Processing {filepath} in chunks...")
        
        file_chunks = []
        try:
            for chunk in pd.read_csv(filepath, chunksize=100000, low_memory=False, on_bad_lines='skip'):
                chunk.columns = chunk.columns.str.strip()
                
                label_col = None
                for col in chunk.columns:
                    if col.lower() in ['label', 'attack_cat', 'target']:
                        label_col = col
                        break
                
                if not label_col:
                    continue

                chunk.rename(columns={label_col: 'label'}, inplace=True)
                chunk['label'] = chunk['label'].astype(str).str.strip()
                chunk = chunk[~chunk['label'].isin(['label', 'nan', 'None', ''])]

                file_chunks.append(chunk)

            if file_chunks:
                df_sub = pd.concat(file_chunks, ignore_index=True)
                
                sub_sampled = []
                for cls_name, cls_df in df_sub.groupby('label'):
                    if cls_name in ALWAYS_KEEP_CLASSES or len(cls_df) <= max_majority_class:
                        sub_sampled.append(cls_df)
                    else:
                        sub_sampled.append(cls_df.sample(n=max_majority_class, random_state=42))

                if sub_sampled:
                    file_result = pd.concat(sub_sampled, ignore_index=True)
                    sampled_chunks.append(file_result)
                    print(f"  -> Reduced {filename} from {len(df_sub)} rows to {len(file_result)} rows.")

                del df_sub, file_chunks, sub_sampled

        except Exception as e:
            print(f"  Warning reading {filename}: {e}")

    if not sampled_chunks:
        print("No valid CICIOT23 data chunks were extracted.")
        return

    print("\nMerging memory-efficient dataset chunks...")
    final_df = pd.concat(sampled_chunks, ignore_index=True)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False)
    
    print(f"\n==================================================")
    print(f"SUCCESSFULLY COMBINED CICIOT23 MASTER DATASET!")
    print(f"Output Saved To: {output_path}")
    print(f"Total Master Dataset Rows: {len(final_df)}")
    print("\nFinal Class Distribution ('label'):")
    print(final_df['label'].value_counts())
    print(f"==================================================")

if __name__ == '__main__':
    combine_ciciot23_files()
