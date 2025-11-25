
"""
1D CNN model builder.
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_cnn(input_shape, filters: int = 64, kernel_size: int = 3, pool_size: int = 2, dropout: float = 0.2, lr: float = 1e-3):
    model = Sequential([
        Conv1D(filters, kernel_size, activation="relu", input_shape=input_shape),
        MaxPooling1D(pool_size=pool_size),
        Flatten(),
        Dropout(dropout),
        Dense(64, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss="mse")
    return model
