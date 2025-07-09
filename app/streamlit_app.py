import streamlit as st
import pandas as pd
import numpy as np
import torch
from utils.preprocessing import TimeSeriesPreprocessor
from models.transformer import TimeSeriesTransformer
from models.glassbox import GlassboxExplainer
from utils.visualization import plot_time_series
from utils.evaluation import Evaluator
from data.synthetic_generator import generate_synthetic_variants

st.title("HealthTimeSeriesAI: Time Series Analysis for Healthcare")

# Data Upload
st.header("Upload Time Series Data")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:")
    st.dataframe(df.head())

    feature_cols = [str(i) for i in range(1, 188)]
    label_col = 'Label'
    preprocessor = TimeSeriesPreprocessor(window_size=187)
    X, y = preprocessor.preprocess(df, feature_cols, label_col)

    # Visualize Data
    st.header("Visualize Time Series")
    if st.button("Plot Time Series"):
        plot_time_series(df, feature_cols, 'time_series.png')
        st.image('time_series.png')

    # Generate Synthetic Data
    st.header("Generate Synthetic Data")
    n_samples = st.slider("Number of synthetic samples", 10, 100, 50)
    if st.button("Generate Synthetic Data"):
        generate_synthetic_variants(uploaded_file, 'data/processed/synthetic_data.csv', n_samples)
        synthetic_df = pd.read_csv('data/processed/synthetic_data.csv')
        st.write("Synthetic Data Preview:")
        st.dataframe(synthetic_df.head())

    # Train Model
    st.header("Train Transformer Model")
    if st.button("Train Model"):
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
            st.write(f"Epoch {epoch}, Loss: {loss.item()}")
        torch.save(model.state_dict(), 'models/transformer_checkpoint.pth')
        st.success("Model trained and saved!")

    # Predict and Explain
    st.header("Predict and Explain")
    if st.button("Run Prediction"):
        model = TimeSeriesTransformer(input_dim=1, n_classes=2)
        model.load_state_dict(torch.load('models/transformer_checkpoint.pth'))
        model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            predictions = model(X_tensor).argmax(dim=1).numpy()
        evaluator = Evaluator(task='classification')
        metrics = evaluator.evaluate(y, predictions)
        st.write("Evaluation Metrics:", metrics)

        # Explain Predictions
        explainer = GlassboxExplainer(model)
        attention = explainer.explain(X_tensor[:1])
        explainer.visualize_attention(attention, feature_cols, 'attention.png')
        st.image('attention.png', caption="Attention Weights")
