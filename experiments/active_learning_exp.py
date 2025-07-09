import torch
import pandas as pd
import numpy as np
from models.active_learning import ActiveLearner
from models.transformer import TimeSeriesTransformer
from utils.preprocessing import TimeSeriesPreprocessor
from utils.evaluation import Evaluator

def active_learning_experiment(data_path='data/raw/sample_dataset.csv', n_labeled=10, n_iterations=5):
    """
    Conduct an active learning experiment, selecting informative samples for labeling.
    """
    # Load and preprocess data
    df = pd.read_csv(data_path)
    feature_cols = [str(i) for i in range(1, 188)]  # Columns 1 to 187
    preprocessor = TimeSeriesPreprocessor(window_size=187)
    X, y = preprocessor.preprocess(df, feature_cols, 'Label')

    # Split into labeled and unlabeled sets
    np.random.seed(42)
    labeled_indices = np.random.choice(len(X), n_labeled, replace=False)
    unlabeled_indices = np.setdiff1d(np.arange(len(X)), labeled_indices)
    
    X_labeled = X[labeled_indices]
    y_labeled = y[labeled_indices]
    X_unlabeled = X[unlabeled_indices]

    # Initialize model and active learner
    model = TimeSeriesTransformer(input_dim=1, n_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    active_learner = ActiveLearner(model, n_samples=5)

    # Active learning loop
    evaluator = Evaluator(task='classification')
    for iteration in range(n_iterations):
        # Train on labeled data
        model.train()
        X_labeled_tensor = torch.tensor(X_labeled, dtype=torch.float32)
        y_labeled_tensor = torch.tensor(y_labeled, dtype=torch.long)
        for epoch in range(10):
            optimizer.zero_grad()
            output = model(X_labeled_tensor)
            loss = criterion(output, y_labeled_tensor)
            loss.backward()
            optimizer.step()

        # Select informative samples from unlabeled data
        X_unlabeled_tensor = torch.tensor(X_unlabeled, dtype=torch.float32)
        selected_indices = active_learner.select_samples(X_unlabeled_tensor)
        
        # Simulate labeling (in practice, request labels from an oracle)
        new_labeled_indices = unlabeled_indices[selected_indices]
        X_new_labeled = X_unlabeled[selected_indices]
        y_new_labeled = y[new_labeled_indices]  # For demo; real-world would query expert

        # Update labeled and unlabeled sets
        X_labeled = np.concatenate([X_labeled, X_new_labeled], axis=0)
        y_labeled = np.concatenate([y_labeled, y_new_labeled], axis=0)
        unlabeled_indices = np.setdiff1d(unlabeled_indices, new_labeled_indices)
        X_unlabeled = X[unlabeled_indices]

        # Evaluate model
        model.eval()
        with torch.no_grad():
            X_all = torch.tensor(X, dtype=torch.float32)
            predictions = model(X_all).argmax(dim=1).numpy()
            metrics = evaluator.evaluate(y, predictions)
            print(f"Iteration {iteration+1}, Metrics: {metrics}, Labeled Samples: {len(X_labeled)}")

    # Semi-supervised learning: pseudo-label remaining unlabeled data
    if len(X_unlabeled) > 0:
        pseudo_labels = active_learner.semi_supervised_update(X_unlabeled)
        X_labeled = np.concatenate([X_labeled, X_unlabeled], axis=0)
        y_labeled = np.concatenate([y_labeled, pseudo_labels], axis=0)

        # Retrain with pseudo-labels
        model.train()
        X_labeled_tensor = torch.tensor(X_labeled, dtype=torch.float32)
        y_labeled_tensor = torch.tensor(y_labeled, dtype=torch.long)
        for epoch in range(10):
            optimizer.zero_grad()
            output = model(X_labeled_tensor)
            loss = criterion(output, y_labeled_tensor)
            loss.backward()
            optimizer.step()

    # Final evaluation
    model.eval()
    with torch.no_grad():
        X_all = torch.tensor(X, dtype=torch.float32)
        predictions = model(X_all).argmax(dim=1).numpy()
        metrics = evaluator.evaluate(y, predictions)
        print(f"Final Metrics with Active and Semi-Supervised Learning: {metrics}")

    # Save model
    torch.save(model.state_dict(), 'models/active_learning_transformer_checkpoint.pth')
    print("Active learning experiment completed and model saved!")

if __name__ == "__main__":
    active_learning_experiment()
