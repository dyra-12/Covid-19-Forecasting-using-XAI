
"""
MLP for flattened sequence inputs (T*F).
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, InputLayer
from tensorflow.keras.optimizers import Adam

def build_mlp(input_dim: int, hidden: int = 128, dropout: float = 0.2, lr: float = 1e-3):
    model = Sequential([
        InputLayer(input_shape=(input_dim,)),
        Dense(hidden, activation="relu"),
        Dropout(dropout),
        Dense(hidden//2, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss="mse")
    return model
