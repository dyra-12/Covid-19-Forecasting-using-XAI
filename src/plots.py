
"""
Plotting helpers.
"""
import matplotlib.pyplot as plt
import numpy as np

def plot_predictions(y_true, y_pred, title: str = "Predictions vs True"):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    plt.figure(figsize=(10, 5))
    plt.plot(y_true, label="True")
    plt.plot(y_pred, label="Pred")
    plt.title(title)
    plt.xlabel("Samples")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()
