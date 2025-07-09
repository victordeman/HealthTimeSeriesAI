import torch
import pandas as pd
import numpy as np
from models.stream_learner import StreamLearner
from models.transformer import TimeSeriesTransformer
from utils.preprocessing import TimeSeriesPreprocessor
from utils.evaluation import Evaluator

def stream_experiment(data_path='data/raw/sample_dataset.csv', batch_size=32, epochs=10):
    """
    Simulate streaming time series data and update the model in real-time.
    """
    # Load and preprocess data
    df = pd.read_csv(data_path)
    feature_cols = [str(i) for i in range(1, 188)]  # Columns 1 to 187
    preprocessor = TimeSeriesPreprocessor(window_size=187)
    X, y = preprocessor.preprocess(df, feature_cols, 'Label')

    # Initialize model and stream learner
    model = TimeSeriesTransformer(input_dim=1, n_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    stream_learner = StreamLearner(model, optimizer, batch_size=batch_size)

    # Simulate streaming data
    evaluator = Evaluator(task='classification')
    predictions = []
    true_labels = []

    for i in range(len(X)):
        # Simulate incoming data point
        new_data = torch.tensor(X[i:i+1], dtype=torch.float32)
        true_label = y[i]

        # Update model with streaming data
        stream_learner.update(new_data)

        # Evaluate periodically (e.g., every batch_size samples)
        if (i + 1) % batch_size == 0:
            model.eval()
            with torch.no_grad():
                batch_X = torch.tensor(X[max(0, i-batch_size+1):i+1], dtype=torch.float32)
                batch_y = y[max(0, i-batch_size+1):i+1]
                output = model(batch_X).argmax(dim=1).numpy()
                predictions.extend(output)
                true_labels.extend(batch_y)
                metrics = evaluator.evaluate(true_labels, predictions)
                print(f"Stream Step {i+1}, Metrics: {metrics}")

    # Final evaluation
    metrics = evaluator.evaluate(true_labels, predictions)
    print(f"Final Streaming Metrics: {metrics}")

    # Save model
    torch.save(model.state_dict(), 'models/stream_transformer_checkpoint.pth')
    print("Streaming experiment completed and model saved!")

if __name__ == "__main__":
    stream_experiment()
