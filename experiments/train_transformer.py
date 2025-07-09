import torch
import pandas as pd
from models.transformer import TimeSeriesTransformer
from utils.preprocessing import TimeSeriesPreprocessor
from data.synthetic_generator import generate_synthetic_variants

def train():
    df = pd.read_csv('data/raw/sample_dataset.csv')
    feature_cols = [str(i) for i in range(1, 188)]
    preprocessor = TimeSeriesPreprocessor(window_size=187)
    X, y = preprocessor.preprocess(df, feature_cols, 'Label')
    
    generate_synthetic_variants('data/raw/sample_dataset.csv', 'data/processed/synthetic_data.csv', n_samples=100)
    synthetic_df = pd.read_csv('data/processed/synthetic_data.csv')
    X_synthetic, y_synthetic = preprocessor.preprocess(synthetic_df, feature_cols, 'Label')
    
    X = np.concatenate([X, X_synthetic], axis=0)
    y = np.concatenate([y, y_synthetic], axis=0)
    
    model = TimeSeriesTransformer(input_dim=1, n_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")
    
    torch.save(model.state_dict(), 'models/transformer_checkpoint.pth')

if __name__ == "__main__":
    train()
