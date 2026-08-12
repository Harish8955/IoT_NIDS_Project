import os
import glob
import pandas as pd

def combine_csv_files(input_dir="data/cic_ids2018_raw", output_path="data/CIC_IDS2018_combined.csv", max_majority_per_file=3000):
    """
    Chunked, ultra-memory-efficient merger for CSE-CIC-IDS2018 daily CSV files.
    Uses pandas chunking to read large files like 02-20-2018 without memory errors.
    """
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not csv_files:
        print(f"\n[ERROR] No CSV files found in '{input_dir}'.")
        return

    print(f"\nFound {len(csv_files)} daily CSV files in '{input_dir}'. Chunk-merging...")
    sampled_chunks = []

    ALWAYS_KEEP_CLASSES = {
        'SQL Injection', 'Brute Force -Web', 'Brute Force -XSS',
        'DDOS attack-LOIC-UDP', 'DDOS attack-LOIC-HTTP', 'DoS attacks-Slowloris', 'DoS attacks-GoldenEye'
    }

    for idx, filepath in enumerate(csv_files, 1):
        filename = os.path.basename(filepath)
        print(f"[{idx}/{len(csv_files)}] Processing {filename} in chunks...")
        
        file_chunks = []
        try:
            # Read in 100,000 row chunks to avoid any C-level memory errors
            for chunk in pd.read_csv(filepath, chunksize=100000, low_memory=False, on_bad_lines='skip'):
                chunk.columns = chunk.columns.str.strip()
                
                label_col = None
                for col in chunk.columns:
                    if col.lower() == 'label':
                        label_col = col
                        break
                
                if not label_col:
                    continue

                chunk.rename(columns={label_col: 'Label'}, inplace=True)
                chunk['Label'] = chunk['Label'].astype(str).str.strip()
                chunk = chunk[~chunk['Label'].isin(['Label', 'nan', 'None', ''])]

                file_chunks.append(chunk)

            if file_chunks:
                df_sub = pd.concat(file_chunks, ignore_index=True)
                
                # Sample file by class
                sub_sampled = []
                for cls_name, cls_df in df_sub.groupby('Label'):
                    if cls_name in ALWAYS_KEEP_CLASSES or len(cls_df) <= max_majority_per_file:
                        sub_sampled.append(cls_df)
                    else:
                        sub_sampled.append(cls_df.sample(n=max_majority_per_file, random_state=42))

                if sub_sampled:
                    file_result = pd.concat(sub_sampled, ignore_index=True)
                    sampled_chunks.append(file_result)
                    print(f"  -> Reduced {filename} from {len(df_sub)} rows to {len(file_result)} rows.")

                del df_sub, file_chunks, sub_sampled

        except Exception as e:
            print(f"  Warning reading {filename}: {e}")

    if not sampled_chunks:
        print("No valid data chunks were extracted.")
        return

    print("\nMerging memory-efficient dataset chunks...")
    final_df = pd.concat(sampled_chunks, ignore_index=True)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False)
    
    print(f"\n==================================================")
    print(f"SUCCESSFULLY COMBINED MASTER DATASET!")
    print(f"Output Saved To: {output_path}")
    print(f"Total Master Dataset Rows: {len(final_df)}")
    print("\nFinal Class Distribution ('Label'):")
    print(final_df['Label'].value_counts())
    print(f"==================================================")

if __name__ == '__main__':
    combine_csv_files()
