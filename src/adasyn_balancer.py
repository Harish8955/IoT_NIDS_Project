import numpy as np
import pandas as pd
from collections import Counter
from imblearn.over_sampling import SMOTENC, ADASYN, RandomOverSampler

class ADASYNBalancer:
    """
    Categorical-Aware Imbalance Balancer:
    - Automatically identifies categorical vs continuous columns.
    - Uses SMOTENC when categorical features are present to prevent fractional artifact corruption.
    - Constrains oversampling so minority classes are balanced up to a sensible ratio (capped at 40% of majority)
      to prevent synthetic dense clusters and decision boundary disruption.
    """
    def __init__(self, target_col='attack_cat', max_minority_ratio=0.4, random_state=42):
        self.target_col = target_col
        self.max_minority_ratio = max_minority_ratio
        self.random_state = random_state

    def balance_dataset(self, df):
        if self.target_col not in df.columns:
            return df

        df = df.copy()
        y = df[self.target_col]
        X = df.drop(columns=[self.target_col])

        counts = Counter(y)
        majority_count = max(counts.values())
        target_cap = max(50, int(majority_count * self.max_minority_ratio))

        sampling_strategy = {}
        for cls_name, count in counts.items():
            if count < target_cap:
                # Require at least 6 samples to build valid k-NN graphs
                if count < 6:
                    sampling_strategy[cls_name] = max(count, 6)
                else:
                    sampling_strategy[cls_name] = target_cap
            else:
                sampling_strategy[cls_name] = count

        # Detect categorical column indices
        cat_indices = []
        for idx, col in enumerate(X.columns):
            if X[col].dtype == 'object' or str(X[col].dtype).startswith('category'):
                cat_indices.append(idx)

        print(f"  [*] Initial Class Distribution: {dict(counts)}")
        print(f"  [*] Target Sampling Caps: {sampling_strategy}")

        # Choose appropriate balancer based on feature types
        try:
            if len(cat_indices) > 0:
                print(f"  [*] Detected {len(cat_indices)} categorical features. Using SMOTENC.")
                sampler = SMOTENC(
                    categorical_features=cat_indices,
                    sampling_strategy=sampling_strategy,
                    k_neighbors=min(5, min(counts.values()) - 1 if min(counts.values()) > 1 else 1),
                    random_state=self.random_state
                )
            else:
                sampler = ADASYN(
                    sampling_strategy=sampling_strategy,
                    n_neighbors=min(5, min(counts.values()) - 1 if min(counts.values()) > 1 else 1),
                    random_state=self.random_state
                )
            X_res, y_res = sampler.fit_resample(X, y)
        except Exception as e:
            print(f"  [!] Interpolation balancer failed ({e}). Falling back to safe RandomOverSampler.")
            sampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=self.random_state)
            X_res, y_res = sampler.fit_resample(X, y)

        resampled_df = pd.DataFrame(X_res, columns=X.columns)
        resampled_df[self.target_col] = y_res
        print(f"  [*] Post-Balancing Class Distribution: {dict(Counter(y_res))}")
        return resampled_df