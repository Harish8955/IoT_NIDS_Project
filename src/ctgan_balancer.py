import pandas as pd
import numpy as np

class CTGANBalancer:
    def __init__(self, target_col='attack_cat', epochs=10):
        self.target_col = target_col
        self.epochs = epochs

    def balance_dataset(self, df, target_count=None):
        """
        Balances minority classes in the tabular dataset using CTGAN.
        """
        df_balanced = df.copy()
        class_counts = df_balanced[self.target_col].value_counts()
        majority_count = class_counts.max()
        
        if target_count is None:
            # Set target count for minority classes to match 30% of majority class for balanced proportions
            target_count = int(majority_count * 0.3)

        minority_classes = class_counts[class_counts < target_count].index.tolist()
        
        if not minority_classes:
            print("No minority classes below target threshold found. Dataset already balanced.")
            return df_balanced

        print(f"Balancing minority classes using CTGAN: {minority_classes}")

        try:
            from ctgan import CTGAN
            
            for cls in minority_classes:
                current_cnt = class_counts[cls]
                num_to_generate = target_count - current_cnt
                if num_to_generate <= 0:
                    continue

                print(f"Training CTGAN on class '{cls}' (Current: {current_cnt}, Generating: {num_to_generate})...")
                sub_df = df_balanced[df_balanced[self.target_col] == cls]
                
                # Identify discrete columns for CTGAN
                discrete_columns = [
                    col for col in sub_df.columns 
                    if sub_df[col].dtype == 'object' or col == self.target_col
                ]

                ctgan_model = CTGAN(epochs=self.epochs, verbose=False)
                ctgan_model.fit(sub_df, discrete_columns)
                
                synthetic_data = ctgan_model.sample(num_to_generate)
                df_balanced = pd.concat([df_balanced, synthetic_data], ignore_index=True)
                print(f"Synthesized {len(synthetic_data)} rows for '{cls}'.")

        except Exception as e:
            print(f"CTGAN fitting warning: {e}. Falling back to Random Oversampling for minority classes.")
            for cls in minority_classes:
                current_cnt = class_counts[cls]
                num_to_generate = target_count - current_cnt
                if num_to_generate > 0:
                    sub_df = df_balanced[df_balanced[self.target_col] == cls]
                    synthetic_data = sub_df.sample(num_to_generate, replace=True)
                    df_balanced = pd.concat([df_balanced, synthetic_data], ignore_index=True)

        print("Final balanced class distribution:")
        print(df_balanced[self.target_col].value_counts())
        return df_balanced
