
"""
Hybrid model: Conv1D -> BiLSTM -> Dense.
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_hybrid_cnn_bilstm(input_shape, filters: int = 64, kernel_size: int = 3, pool_size: int = 2,
                            lstm_units: int = 64, dropout: float = 0.2, lr: float = 1e-3):
    model = Sequential([
        Conv1D(filters, kernel_size, activation="relu", input_shape=input_shape),
        MaxPooling1D(pool_size=pool_size),
        Bidirectional(LSTM(lstm_units)),
        Dropout(dropout),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss="mse")
    return model
