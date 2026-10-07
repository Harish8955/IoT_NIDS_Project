import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class TrafficDataPreprocessor:
    def __init__(self, target_col='attack_cat'):
        self.target_col = target_col
        self.label_encoders = {}
        self.target_encoder = LabelEncoder()
        self.feature_mins = None
        self.feature_maxs = None
        self.continuous_cols = []
        self.clip_lows = None
        self.clip_highs = None
        self.feature_means = None
        self.feature_scales = None
        self.feature_cols = []
        self.categorical_cols = []
        self.encoded_feature_cols = []

    def _sanitize_matrix(self, X_mat):
        """Replaces NaN, +Inf, -Inf with 0.0"""
        return np.nan_to_num(X_mat, nan=0.0, posinf=0.0, neginf=0.0)

    def fit_transform(self, df, hide_class=None):
        """Preprocesses raw tabular traffic data (Fitted on Training data)."""
        df = df.copy()

        for col_to_drop in ['id', 'ID', 'Id', 'Unnamed: 0']:
            if col_to_drop in df.columns:
                df = df.drop(columns=[col_to_drop])

        if self.target_col == 'attack_cat' and 'label' in df.columns:
            df = df.drop(columns=['label'])

        if self.target_col in df.columns:
            df[self.target_col] = df[self.target_col].astype(str).str.strip()
            # Analysis and Backdoor are valid UNSW-NB15 classes; numeric zero may
            # also be a valid benign label. Remove only missing/header sentinels.
            df = df[~df[self.target_col].isin(['nan', 'None', 'Label', 'label'])]

        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

        zero_day_df = None
        if hide_class and self.target_col in df.columns:
            if hide_class in df[self.target_col].values:
                zero_day_df = df[df[self.target_col] == hide_class].copy()
                df = df[df[self.target_col] != hide_class].copy()
                print(f"[LOCO] ZERO-DAY SIMULATION: Hiding '{hide_class}' ({len(zero_day_df)} samples) from training split!")

        if self.target_col in df.columns:
            y = self.target_encoder.fit_transform(df[self.target_col])
            X_df = df.drop(columns=[self.target_col])
        else:
            y = None
            X_df = df

        self.feature_cols = X_df.columns.tolist()
        self.categorical_cols = [
            col for col in self.feature_cols
            if X_df[col].dtype == 'object' or str(X_df[col].dtype).startswith('category')
        ]
        X_encoded = self._encode_features(X_df, fit=True)
        self.encoded_feature_cols = X_encoded.columns.tolist()
        self.continuous_cols = [c for c in self.encoded_feature_cols if c in self.feature_cols and c not in self.categorical_cols]
        X_norm = self._fit_scale(X_encoded)

        return X_norm, y, zero_day_df

    def transform(self, df):
        """Transform test/validation/zero-day set using fitted training parameters."""
        df = df.copy()

        for col_to_drop in ['id', 'ID', 'Id', 'Unnamed: 0']:
            if col_to_drop in df.columns:
                df = df.drop(columns=[col_to_drop])

        if self.target_col == 'attack_cat' and 'label' in df.columns:
            df = df.drop(columns=['label'])

        if self.target_col in df.columns:
            df[self.target_col] = df[self.target_col].astype(str).str.strip()
            df = df[~df[self.target_col].isin(['nan', 'None', 'Label', 'label'])]

        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

        if self.target_col in df.columns:
            valid_classes = set(self.target_encoder.classes_)
            y = np.array([-1] * len(df))
            valid_mask = df[self.target_col].isin(valid_classes)
            if valid_mask.any():
                y[valid_mask] = self.target_encoder.transform(df[self.target_col][valid_mask])
            X_df = df.drop(columns=[self.target_col])
        else:
            y = np.array([-1] * len(df))
            X_df = df

        X_encoded = self._encode_features(X_df, fit=False)

        X_norm = self._scale(X_encoded)

        return X_norm, y

    def _encode_features(self, X_df, fit):
        """One-hot encode categorical traffic fields using training columns only."""
        frame = X_df.reindex(columns=self.feature_cols).copy()
        for col in self.feature_cols:
            if col not in self.categorical_cols:
                frame[col] = pd.to_numeric(frame[col], errors='coerce').replace(
                    [np.inf, -np.inf], np.nan
                ).fillna(0)
            else:
                frame[col] = frame[col].fillna('__missing__').astype(str)
        encoded = pd.get_dummies(frame, columns=self.categorical_cols, dtype=np.float32)
        if fit:
            return encoded
        return encoded.reindex(columns=self.encoded_feature_cols, fill_value=0)

    def _fit_scale(self, X_encoded):
        """Clip and standardize continuous fields using training rows only."""
        values = X_encoded.to_numpy(dtype=np.float32, copy=True)
        self.feature_mins = np.min(values, axis=0)
        self.feature_maxs = np.max(values, axis=0)
        self.clip_lows = np.full(values.shape[1], -np.inf, dtype=np.float32)
        self.clip_highs = np.full(values.shape[1], np.inf, dtype=np.float32)
        self.feature_means = np.zeros(values.shape[1], dtype=np.float32)
        self.feature_scales = np.ones(values.shape[1], dtype=np.float32)
        for col in self.continuous_cols:
            i = self.encoded_feature_cols.index(col)
            low, high = np.quantile(values[:, i], [0.005, 0.995])
            if low == high:
                low, high = float(values[:, i].min()), float(values[:, i].max())
            self.clip_lows[i], self.clip_highs[i] = low, high
            clipped = np.clip(values[:, i], low, high)
            mean = float(clipped.mean())
            scale = float(clipped.std())
            self.feature_means[i] = mean
            self.feature_scales[i] = scale if np.isfinite(scale) and scale > 1e-8 else 1.0
        return self._scale(X_encoded)

    def _scale(self, X_encoded):
        values = self._sanitize_matrix(X_encoded.to_numpy(dtype=np.float32, copy=True))
        for col in self.continuous_cols:
            i = self.encoded_feature_cols.index(col)
            values[:, i] = np.clip(values[:, i], self.clip_lows[i], self.clip_highs[i])
            values[:, i] = (values[:, i] - self.feature_means[i]) / self.feature_scales[i]
        return self._sanitize_matrix(values.astype(np.float32))

def prepare_train_test_split(df, target_col='attack_cat', test_size=0.2, random_state=42, hide_class=None):
    df = df.copy()
    zero_day_df = None
    if hide_class and target_col in df.columns:
        mask = df[target_col].astype(str).str.strip().str.lower() == hide_class.lower()
        zero_day_df = df.loc[mask].copy()
        df = df.loc[~mask].copy()
    label_text = df[target_col].astype(str).str.strip()
    valid = ~label_text.str.lower().isin(['nan', 'none', 'label']) & df[target_col].notna()
    df = df.loc[valid].copy()
    counts = df[target_col].value_counts()
    if len(counts) and counts.min() < 2:
        raise ValueError('Stratified train/test splitting needs at least two examples per class.')
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[target_col]
    )
    preprocessor = TrafficDataPreprocessor(target_col=target_col)
    X_train, y_train, _ = preprocessor.fit_transform(train_df)
    X_test, y_test = preprocessor.transform(test_df)
    return X_train, X_test, y_train, y_test, preprocessor, zero_day_df

def prepare_official_split(train_df, test_df, target_col='attack_cat', hide_class=None):
    preprocessor = TrafficDataPreprocessor(target_col=target_col)
    X_train, y_train, zero_day_df = preprocessor.fit_transform(train_df, hide_class=hide_class)
    X_test, y_test = preprocessor.transform(test_df)
    return X_train, X_test, y_train, y_test, preprocessor, zero_day_df
