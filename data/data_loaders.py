import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from utils.preprocessing import TimeSeriesPreprocessor

class TimeSeriesDataset(Dataset):
    """Custom Dataset for multivariate time series data."""
    def __init__(self, data_path, feature_cols, label_col=None, window_size=187):
        """
        Initialize dataset.
        
        Args:
            data_path (str): Path to CSV file (e.g., 'data/raw/sample_dataset.csv').
            feature_cols (list): List of column names for features (e.g., ['1', '2', ..., '187']).
            label_col (str, optional): Name of label column (e.g., 'Label').
            window_size (int): Length of time series sequence (default: 187).
        """
        self.df = pd.read_csv(data_path)
        self.preprocessor = TimeSeriesPreprocessor(window_size=window_size)
        if label_col:
            self.X, self.y = self.preprocessor.preprocess(self.df, feature_cols, label_col)
            self.y = torch.tensor(self.y, dtype=torch.long)
        else:
            self.X = self.preprocessor.preprocess(self.df, feature_cols)
            self.y = None
        self.X = torch.tensor(self.X, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

def get_data_loader(data_path, feature_cols, label_col='Label', batch_size=32, shuffle=True, window_size=187):
    """
    Create a PyTorch DataLoader for time series data.
    
    Args:
        data_path (str): Path to CSV file.
        feature_cols (list): List of feature column names.
        label_col (str, optional): Name of label column.
        batch_size (int): Batch size for DataLoader.
        shuffle (bool): Whether to shuffle the data.
        window_size (int): Length of time series sequence.
    
    Returns:
        DataLoader: PyTorch DataLoader for the dataset.
    """
    dataset = TimeSeriesDataset(data_path, feature_cols, label_col, window_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def get_unlabeled_data_loader(data_path, feature_cols, batch_size=32, shuffle=False, window_size=187):
    """
    Create a DataLoader for unlabeled data (e.g., for active learning).
    
    Args:
        data_path (str): Path to CSV file.
        feature_cols (list): List of feature column names.
        batch_size (int): Batch size for DataLoader.
        shuffle (bool): Whether to shuffle the data.
        window_size (int): Length of time series sequence.
    
    Returns:
        DataLoader: PyTorch DataLoader for unlabeled data.
    """
    dataset = TimeSeriesDataset(data_path, feature_cols, label_col=None, window_size=window_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

if __name__ == "__main__":
    # Example usage
    data_path = 'data/raw/sample_dataset.csv'
    feature_cols = [str(i) for i in range(1, 188)]
    loader = get_data_loader(data_path, feature_cols, label_col='Label', batch_size=32)
    for batch_X, batch_y in loader:
        print(f"Batch X shape: {batch_X.shape}, Batch y shape: {batch_y.shape}")
        break
