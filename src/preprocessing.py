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

    def _sanitize_matrix(self, X_mat):
        """Replaces NaN, +Inf, -Inf with 0.0"""
        return np.nan_to_num(X_mat, nan=0.0, posinf=0.0, neginf=0.0)

    def fit_transform(self, df, hide_class=None):
        """
        Preprocesses raw tabular traffic data.
        """
        df = df.copy()

        for col_to_drop in ['id', 'ID', 'Id', 'Unnamed: 0']:
            if col_to_drop in df.columns:
                df = df.drop(columns=[col_to_drop])

        if self.target_col == 'attack_cat' and 'label' in df.columns:
            df = df.drop(columns=['label'])

        if self.target_col in df.columns:
            df[self.target_col] = df[self.target_col].astype(str).str.strip()
            df = df[~df[self.target_col].isin(['Analysis', 'Backdoor', 'Backdoors', 'nan', '0', '0.0', 'Label', 'label'])]

        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

        zero_day_df = None
        if hide_class and self.target_col in df.columns:
            if hide_class in df[self.target_col].values:
                zero_day_df = df[df[self.target_col] == hide_class].copy()
                df = df[df[self.target_col] != hide_class].copy()
                print(f"⚠️ LOCO ZERO-DAY SIMULATION: Hiding '{hide_class}' ({len(zero_day_df)} samples) from training split!")

        if self.target_col in df.columns:
            y = self.target_encoder.fit_transform(df[self.target_col])
            X_df = df.drop(columns=[self.target_col])
        else:
            y = None
            X_df = df

        self.feature_cols = X_df.columns.tolist()

        X_encoded = pd.DataFrame()
        for col in self.feature_cols:
            if X_df[col].dtype == 'object' or (len(X_df[col]) > 0 and isinstance(X_df[col].iloc[0], str)):
                # Try numeric conversion first in case numbers are formatted as strings
                num_conv = pd.to_numeric(X_df[col], errors='coerce')
                if num_conv.notna().sum() > (0.5 * len(X_df[col])):
                    X_encoded[col] = num_conv.replace([np.inf, -np.inf], np.nan).fillna(0)
                else:
                    le = LabelEncoder()
                    X_encoded[col] = le.fit_transform(X_df[col].astype(str))
                    self.label_encoders[col] = le
            else:
                X_encoded[col] = pd.to_numeric(X_df[col], errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0)

        X_mat = self._sanitize_matrix(X_encoded.values.astype(np.float32))

        self.feature_mins = np.min(X_mat, axis=0)
        self.feature_maxs = np.max(X_mat, axis=0)
        
        denom = (self.feature_maxs - self.feature_mins)
        denom[denom == 0] = 1.0
        denom[np.isnan(denom)] = 1.0
        denom[np.isinf(denom)] = 1.0

        X_norm = ((X_mat - self.feature_mins) / denom) * 255.0
        X_norm = self._sanitize_matrix(X_norm)

        return X_norm, y, zero_day_df

    def transform(self, df):
        """Transform test set using fitted training parameters."""
        df = df.copy()

        for col_to_drop in ['id', 'ID', 'Id', 'Unnamed: 0']:
            if col_to_drop in df.columns:
                df = df.drop(columns=[col_to_drop])

        if self.target_col == 'attack_cat' and 'label' in df.columns:
            df = df.drop(columns=['label'])

        if self.target_col in df.columns:
            df[self.target_col] = df[self.target_col].astype(str).str.strip()
            df = df[~df[self.target_col].isin(['Analysis', 'Backdoor', 'Backdoors', 'nan', '0', '0.0', 'Label', 'label'])]

        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

        if self.target_col in df.columns:
            valid_classes = set(self.target_encoder.classes_)
            df = df[df[self.target_col].isin(valid_classes)]
            y = self.target_encoder.transform(df[self.target_col])
            X_df = df.drop(columns=[self.target_col])
        else:
            y = None
            X_df = df

        X_encoded = pd.DataFrame()
        for col in self.feature_cols:
            if col in X_df.columns:
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    X_encoded[col] = X_df[col].astype(str).map(
                        lambda s: le.transform([s])[0] if s in le.classes_ else 0
                    )
                else:
                    X_encoded[col] = pd.to_numeric(X_df[col], errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0)
            else:
                X_encoded[col] = 0

        X_mat = self._sanitize_matrix(X_encoded.values.astype(np.float32))
        
        denom = (self.feature_maxs - self.feature_mins)
        denom[denom == 0] = 1.0
        denom[np.isnan(denom)] = 1.0
        denom[np.isinf(denom)] = 1.0

        X_norm = ((X_mat - self.feature_mins) / denom) * 255.0
        X_norm = self._sanitize_matrix(X_norm)

        return X_norm, y

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
