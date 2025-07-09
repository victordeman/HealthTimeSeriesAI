import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class TimeSeriesPreprocessor:
    def __init__(self, window_size=187, stride=1):
        self.window_size = window_size
        self.stride = stride
        self.scaler = StandardScaler()

    def preprocess(self, df, feature_cols, label_col=None):
        """Preprocess multivariate time series data."""
        df[feature_cols] = df[feature_cols].fillna(method='ffill').fillna(method='bfill')
        X = self.scaler.fit_transform(df[feature_cols])
        windows = X.reshape(-1, self.window_size, len(feature_cols))
        if label_col:
            y = df[label_col].values
            return windows, y
        return windows

    def inverse_transform(self, data):
        """Inverse transform normalized data."""
        return self.scaler.inverse_transform(data)
