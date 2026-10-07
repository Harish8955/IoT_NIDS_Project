import numpy as np
import pandas as pd
from collections import Counter
from imblearn.over_sampling import SMOTE, SMOTENC, ADASYN, RandomOverSampler

class AdaptiveClassBalancer:
    """
    Robust Class Imbalance Balancer for Tabular NIDS Datasets.
    
    Dispatch Order:
    1. If user forces 'random' or minimum class count < 2 -> RandomOverSampler.
    2. If dataset contains categorical features:
       - Uses SMOTENC with categorical indices identified to prevent numerical interpolation corruption.
    3. If dataset is continuous:
       - Uses ADASYN (focusing on minority density / decision boundary hardness).
    4. If ADASYN cannot synthesize a requested class, retry that class with SMOTE.
       RandomOverSampler remains available for explicit requests or when interpolation
       is impossible because there are too few samples.
    """
    def __init__(self, target_col='attack_cat', max_minority_ratio=0.1, random_state=42, preferred_sampler='auto'):
        self.target_col = target_col
        self.max_minority_ratio = max_minority_ratio
        self.random_state = random_state
        self.preferred_sampler = str(preferred_sampler).lower()
        self.actual_methods = {}

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
        target_strategy = {
            cls_name: target_count
            for cls_name, target_count in sampling_strategy.items()
            if target_count > initial_counts[cls_name]
        }

        cat_indices = [
            i for i, col in enumerate(X.columns)
            if X[col].dtype == 'object' or str(X[col].dtype).startswith('category')
        ]
        has_categorical = len(cat_indices) > 0
        safe_k = min(5, max(1, min_class_samples - 1))
        can_interpolate = min_class_samples > 1

        print(f"  [*] Initial Class Distribution: {dict(initial_counts)}")
        print(f"  [*] Target Strategy (capped at {int(self.max_minority_ratio * 100)}% of majority): {sampling_strategy}")

        # Only classes that actually need more samples are passed to samplers.
        if not target_strategy:
            print("  [*] No classes require oversampling; leaving training partition unchanged.")
            self.actual_methods = {"all": "none (no classes under the cap)"}
            return df

        # Explicit Dispatch
        sampler = None
        sampler_name = ""

        if self.preferred_sampler == 'random' or not can_interpolate:
            sampler_name = "RandomOverSampler (Forced or Insufficient Neighbor Support)"
            sampler = RandomOverSampler(sampling_strategy=target_strategy, random_state=self.random_state)

        elif has_categorical:
            if self.preferred_sampler == 'adasyn':
                print("  [!] Warning: ADASYN requested but categorical columns detected. Switching to SMOTENC to avoid corruption.")
            sampler_name = f"SMOTENC (Categorical-Aware, k={safe_k})"
            sampler = SMOTENC(
                categorical_features=cat_indices,
                sampling_strategy=target_strategy,
                k_neighbors=safe_k,
                random_state=self.random_state
            )

        else:  # Purely continuous data
            if self.preferred_sampler == 'smotenc':
                print("  [!] Warning: SMOTENC requested on continuous data. Dispatching ADASYN.")
            sampler_name = f"ADASYN (Continuous Density, n={safe_k})"
            sampler = ADASYN(
                sampling_strategy=target_strategy,
                n_neighbors=safe_k,
                random_state=self.random_state
            )

        print(f"  [*] Selected Balancer: {sampler_name}")

        actual_methods = {}
        try:
            X_res, y_res = sampler.fit_resample(X, y)
            for cls_name in target_strategy:
                actual_methods[cls_name] = sampler_name
        except Exception as e:
            print(f"  [!] {sampler_name} failed for the complete strategy ({e}).")
            if sampler_name.startswith("ADASYN"):
                # ADASYN can assign zero synthetic samples to an easy class and
                # raise, cancelling resampling for every other requested class.
                # Retry classes independently so only unsupported classes need
                # a non-synthetic fallback.
                X_res, y_res = X, y
                for cls_name, target_count in target_strategy.items():
                    class_strategy = {cls_name: target_count}
                    try:
                        class_sampler = ADASYN(
                            sampling_strategy=class_strategy,
                            n_neighbors=safe_k,
                            random_state=self.random_state
                        )
                        X_res, y_res = class_sampler.fit_resample(X_res, y_res)
                        actual_methods[cls_name] = "ADASYN"
                    except Exception as class_error:
                        print(
                            f"  [!] ADASYN could not synthesize class {cls_name} "
                            f"({class_error}); retrying this class with SMOTE."
                        )
                        try:
                            fallback = SMOTE(
                                sampling_strategy=class_strategy,
                                k_neighbors=min(safe_k, int(initial_counts[cls_name]) - 1),
                                random_state=self.random_state
                            )
                            X_res, y_res = fallback.fit_resample(X_res, y_res)
                            actual_methods[cls_name] = "SMOTE fallback"
                        except Exception as smote_error:
                            # Avoid silently repeating a class many times when both
                            # synthetic methods fail; weighted loss still uses originals.
                            print(
                                f"  [!] SMOTE could not synthesize class {cls_name} "
                                f"({smote_error}); leaving that class at its original count."
                            )
                            actual_methods[cls_name] = "no oversampling (ADASYN/SMOTE unavailable)"
            else:
                print("  [!] Falling back to RandomOverSampler for the requested classes.")
                fallback = RandomOverSampler(
                    sampling_strategy=target_strategy,
                    random_state=self.random_state
                )
                X_res, y_res = fallback.fit_resample(X, y)
                for cls_name in target_strategy:
                    actual_methods[cls_name] = "RandomOverSampler fallback"

        methods_used = sorted(set(actual_methods.values()))
        self.actual_methods = actual_methods
        print(f"  [*] Actual Balancing Method(s): {', '.join(methods_used)}")
        if "RandomOverSampler fallback" in methods_used:
            fallback_classes = [
                str(cls_name) for cls_name, method in actual_methods.items()
                if method == "RandomOverSampler fallback"
            ]
            print(f"  [!] RandomOverSampler was used only for classes: {fallback_classes}")

        resampled_df = pd.DataFrame(np.asarray(X_res), columns=X.columns)
        resampled_df = pd.concat(
            [resampled_df, pd.Series(np.asarray(y_res), name=self.target_col)],
            axis=1
        )
        resampled_df.attrs["balancing_methods"] = actual_methods
        print(f"  [*] Balanced Class Distribution: {dict(Counter(y_res))}")
        return resampled_df

# Alias for backwards compatibility
ADASYNBalancer = AdaptiveClassBalancer
