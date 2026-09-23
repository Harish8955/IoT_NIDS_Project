import numpy as np
import pandas as pd
from collections import Counter
from imblearn.over_sampling import SMOTENC, ADASYN, RandomOverSampler

class AdaptiveClassBalancer:
    """
    Polymorphic Class Imbalance Balancer for Tabular NIDS Datasets.
    
    Balancing Hierarchy:
    1. Mixed / Categorical data present -> SMOTENC (prevents fractional feature corruption)
    2. Purely continuous data           -> ADASYN (focuses on boundary hardness)
    3. Low sample count / degenerate kNN -> RandomOverSampler (robust fallback)
    """
    def __init__(self, target_col='attack_cat', max_minority_ratio=0.4, random_state=42, preferred_sampler='auto'):
        self.target_col = target_col
        self.max_minority_ratio = max_minority_ratio
        self.random_state = random_state
        self.preferred_sampler = preferred_sampler.lower()  # 'auto', 'smotenc', 'adasyn', 'random'

    def _compute_sampling_caps(self, y):
        counts = Counter(y)
        majority_count = max(counts.values())
        target_cap = max(50, int(majority_count * self.max_minority_ratio))

        strategy = {}
        for cls_name, count in counts.items():
            if count < target_cap:
                strategy[cls_name] = max(count, 6) if count < 6 else target_cap
            else:
                strategy[cls_name] = count
        return counts, strategy

    def balance_dataset(self, df):
        if self.target_col not in df.columns:
            return df

        df = df.copy()
        y = df[self.target_col]
        X = df.drop(columns=[self.target_col])

        initial_counts, sampling_strategy = self._compute_sampling_caps(y)
        min_class_samples = min(initial_counts.values())
        k_neighbors = min(5, max(1, min_class_samples - 1))

        # Identify categorical column indices
        cat_indices = [
            i for i, col in enumerate(X.columns)
            if X[col].dtype == 'object' or str(X[col].dtype).startswith('category')
        ]

        print(f"  [*] Initial Class Distribution: {dict(initial_counts)}")
        print(f"  [*] Target Sampling Strategy (capped at {int(self.max_minority_ratio*100)}% of majority): {sampling_strategy}")

        sampler = None
        sampler_name = ""

        # Dispatch based on data modalities & configuration
        if len(cat_indices) > 0 or self.preferred_sampler == 'smotenc':
            sampler_name = "SMOTENC (Categorical-Aware)"
            sampler = SMOTENC(
                categorical_features=cat_indices,
                sampling_strategy=sampling_strategy,
                k_neighbors=k_neighbors,
                random_state=self.random_state
            )
        elif self.preferred_sampler == 'adasyn' or len(cat_indices) == 0:
            sampler_name = "ADASYN (Continuous Density)"
            sampler = ADASYN(
                sampling_strategy=sampling_strategy,
                n_neighbors=k_neighbors,
                random_state=self.random_state
            )
        else:
            sampler_name = "RandomOverSampler"
            sampler = RandomOverSampler(
                sampling_strategy=sampling_strategy,
                random_state=self.random_state
            )

        print(f"  [*] Selected Balancing Engine: {sampler_name}")

        try:
            X_res, y_res = sampler.fit_resample(X, y)
        except Exception as e:
            print(f"  [!] Primary sampler ({sampler_name}) failed: {e}")
            print("  [*] Falling back to safe RandomOverSampler.")
            fallback = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=self.random_state)
            X_res, y_res = fallback.fit_resample(X, y)

        resampled_df = pd.DataFrame(X_res, columns=X.columns)
        resampled_df[self.target_col] = y_res
        print(f"  [*] Balanced Class Distribution: {dict(Counter(y_res))}")
        return resampled_df


# Backward-compatible alias so existing imports in train.py don't break immediately
ADASYNBalancer = AdaptiveClassBalancer