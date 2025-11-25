
"""
Preprocessing utilities: scaling, sequence creation, and dataset splits.
"""
from typing import Tuple, Optional, Dict
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

def scale_features(df: pd.DataFrame, feature_cols, scaler_type: str = "minmax"):
    """
    Fit a scaler on feature_cols and transform them. Returns scaled array and the fitted scaler.
    scaler_type: "minmax" or "standard"
    """
    if scaler_type == "minmax":
        scaler = MinMaxScaler()
    elif scaler_type == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError("scaler_type must be 'minmax' or 'standard'")
    X = df[feature_cols].values
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def create_sequences(data: np.ndarray, targets: np.ndarray, time_steps: int = 14) -> Tuple[np.ndarray, np.ndarray]:
    """
    Turn a (N, F) data matrix into sequences (N - time_steps, time_steps, F) and aligned targets.
    """
    Xs, ys = [], []
    for i in range(len(data) - time_steps):
        Xs.append(data[i:i+time_steps])
        ys.append(targets[i+time_steps])
    return np.array(Xs), np.array(ys)

def train_test_sequence_split(
    X_seq: np.ndarray, y_seq: np.ndarray, test_size: float = 0.2, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return train_test_split(X_seq, y_seq, test_size=test_size, random_state=random_state, shuffle=False)
