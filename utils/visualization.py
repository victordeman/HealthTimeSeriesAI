import matplotlib.pyplot as plt
import pandas as pd

def plot_time_series(df, feature_cols, filename='time_series.png'):
    plt.figure(figsize=(12, 6))
    for col in feature_cols[:min(3, len(feature_cols))]:  # Plot up to 3 features
        plt.plot(df[col], label=col)
    plt.title("Time Series Data")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig(filename)
    plt.close()
