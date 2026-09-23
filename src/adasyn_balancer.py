import numpy as np
import pandas as pd
from collections import Counter
from imblearn.over_sampling import SMOTENC, ADASYN, RandomOverSampler

class AdaptiveClassBalancer:
    """
    Robust Class Imbalance Balancer for Tabular NIDS Datasets.
    
    Dispatch Order:
    1. If user forces 'random' or minimum class count < 2 -> RandomOverSampler.
    2. If dataset contains categorical features:
       - Uses SMOTENC with categorical indices identified to prevent numerical interpolation corruption.
    3. If dataset is continuous:
       - Uses ADASYN (focusing on minority density / decision boundary hardness).
    4. Any runtime exception -> Graceful fallback to RandomOverSampler.
    """
    def __init__(self, target_col='attack_cat', max_minority_ratio=0.4, random_state=42, preferred_sampler='auto'):
        self.target_col = target_col
        self.max_minority_ratio = max_minority_ratio
        self.random_state = random_state
        self.preferred_sampler = str(preferred_sampler).lower()

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

        cat_indices = [
            i for i, col in enumerate(X.columns)
            if X[col].dtype == 'object' or str(X[col].dtype).startswith('category')
        ]
        has_categorical = len(cat_indices) > 0
        safe_k = min(5, max(1, min_class_samples - 1))
        can_interpolate = min_class_samples > 1

        print(f"  [*] Initial Class Distribution: {dict(initial_counts)}")
        print(f"  [*] Target Strategy (capped at {int(self.max_minority_ratio * 100)}% of majority): {sampling_strategy}")

        # Explicit Dispatch
        sampler = None
        sampler_name = ""

        if self.preferred_sampler == 'random' or not can_interpolate:
            sampler_name = "RandomOverSampler (Forced or Insufficient Neighbor Support)"
            sampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=self.random_state)

        elif has_categorical:
            if self.preferred_sampler == 'adasyn':
                print("  [!] Warning: ADASYN requested but categorical columns detected. Switching to SMOTENC to avoid corruption.")
            sampler_name = f"SMOTENC (Categorical-Aware, k={safe_k})"
            sampler = SMOTENC(
                categorical_features=cat_indices,
                sampling_strategy=sampling_strategy,
                k_neighbors=safe_k,
                random_state=self.random_state
            )

        else:  # Purely continuous data
            if self.preferred_sampler == 'smotenc':
                print("  [!] Warning: SMOTENC requested on continuous data. Dispatching ADASYN.")
            sampler_name = f"ADASYN (Continuous Density, n={safe_k})"
            sampler = ADASYN(
                sampling_strategy=sampling_strategy,
                n_neighbors=safe_k,
                random_state=self.random_state
            )

        print(f"  [*] Selected Balancer: {sampler_name}")

        try:
            X_res, y_res = sampler.fit_resample(X, y)
        except Exception as e:
            print(f"  [!] {sampler_name} failed ({e}). Falling back to RandomOverSampler.")
            fallback = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=self.random_state)
            X_res, y_res = fallback.fit_resample(X, y)

        resampled_df = pd.DataFrame(X_res, columns=X.columns)
        resampled_df[self.target_col] = y_res
        print(f"  [*] Balanced Class Distribution: {dict(Counter(y_res))}")
        return resampled_df

# Alias for backwards compatibility
ADASYNBalancer = AdaptiveClassBalancer