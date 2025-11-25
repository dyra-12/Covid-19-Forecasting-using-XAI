
"""
Example training script showing how modules fit together.
"""
from pathlib import Path
import numpy as np
from xai_pipeline.data_loader import load_covid_aggregated
from xai_pipeline.preprocessing import scale_features, create_sequences, train_test_sequence_split
from xai_pipeline.models.lstm import build_lstm
from xai_pipeline.metrics import regression_metrics
from xai_pipeline.plots import plot_predictions
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def main(csv_path: str):
    df = load_covid_aggregated(csv_path)
    features = ["Confirmed", "Deaths", "Recovered", "Active"]
    target_col = "Confirmed"
    X_scaled, scaler = scale_features(df, features, scaler_type="minmax")
    y = df[target_col].values
    X_seq, y_seq = create_sequences(X_scaled, y, time_steps=14)
    X_train, X_test, y_train, y_test = train_test_sequence_split(X_seq, y_seq, test_size=0.2)
    model = build_lstm(input_shape=X_train.shape[1:])
    callbacks = [EarlyStopping(patience=10, restore_best_weights=True),
                 ReduceLROnPlateau(patience=5)]
    model.fit(X_train, y_train, validation_split=0.1, epochs=50, batch_size=32, callbacks=callbacks, verbose=0)
    y_pred = model.predict(X_test)
    print(regression_metrics(y_test, y_pred))
    plot_predictions(y_test, y_pred, title="LSTM - Test Set")
    return model, X_test, y_test, y_pred

if __name__ == "__main__":
    # Replace with your path if needed
    default_csv = "../data/covid_19_data.csv"
    if Path(default_csv).exists():
        main(default_csv)
    else:
        print("Please provide a valid CSV path to run the example.")
