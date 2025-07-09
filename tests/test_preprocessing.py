import unittest
import pandas as pd
import numpy as np
from utils.preprocessing import TimeSeriesPreprocessor

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        """Set up test cases."""
        self.preprocessor = TimeSeriesPreprocessor(window_size=187)
        self.data = pd.DataFrame({
            '1': [0.5, 0.3, np.nan, 0.7, 0.2],
            '2': [0.4, 0.2, 0.1, 0.6, 0.3],
            'Label': [0, 1, 0, 1, 0]
        })
        self.feature_cols = ['1', '2']

    def test_preprocess_labeled(self):
        """Test preprocessing with labeled data."""
        X, y = self.preprocessor.preprocess(self.data, self.feature_cols, 'Label')
        self.assertEqual(X.shape, (5, 187, 2))  # (samples, window_size, features)
        self.assertEqual(y.shape, (5,))  # (samples,)
        self.assertFalse(np.any(np.isnan(X)))  # No missing values

    def test_preprocess_unlabeled(self):
        """Test preprocessing without labels."""
        X = self.preprocessor.preprocess(self.data, self.feature_cols)
        self.assertEqual(X.shape, (5, 187, 2))  # (samples, window_size, features)
        self.assertFalse(np.any(np.isnan(X)))  # No missing values

    def test_inverse_transform(self):
        """Test inverse transform."""
        X, _ = self.preprocessor.preprocess(self.data, self.feature_cols, 'Label')
        X_inv = self.preprocessor.inverse_transform(X.reshape(-1, 2)).reshape(X.shape)
        self.assertEqual(X_inv.shape, X.shape)

if __name__ == '__main__':
    unittest.main()
