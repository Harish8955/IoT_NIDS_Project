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

        X_mat = self._sanitize_matrix(X_encoded.values.astype(np.float32))

        self.feature_mins = np.min(X_mat, axis=0)
        self.feature_maxs = np.max(X_mat, axis=0)
        
        denom = (self.feature_maxs - self.feature_mins)
        denom[denom == 0] = 1.0
        denom[np.isnan(denom)] = 1.0
        denom[np.isinf(denom)] = 1.0

        # Normalization to [0, 1] for neural network convergence
        X_norm = (X_mat - self.feature_mins) / denom
        X_norm = self._sanitize_matrix(X_norm)

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

        X_mat = self._sanitize_matrix(X_encoded.values.astype(np.float32))
        
        denom = (self.feature_maxs - self.feature_mins)
        denom[denom == 0] = 1.0
        denom[np.isnan(denom)] = 1.0
        denom[np.isinf(denom)] = 1.0

        # Normalization to [0, 1]
        X_norm = (X_mat - self.feature_mins) / denom
        X_norm = self._sanitize_matrix(X_norm)

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

def prepare_train_test_split(df, target_col='attack_cat', test_size=0.2, random_state=42, hide_class=None):
    preprocessor = TrafficDataPreprocessor(target_col=target_col)
    X_norm, y, zero_day_df = preprocessor.fit_transform(df, hide_class=hide_class)
    X_train, X_test, y_train, y_test = train_test_split(
        X_norm, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, preprocessor, zero_day_df

def prepare_official_split(train_df, test_df, target_col='attack_cat', hide_class=None):
    preprocessor = TrafficDataPreprocessor(target_col=target_col)
    X_train, y_train, zero_day_df = preprocessor.fit_transform(train_df, hide_class=hide_class)
    X_test, y_test = preprocessor.transform(test_df)
    return X_train, X_test, y_train, y_test, preprocessor, zero_day_df
