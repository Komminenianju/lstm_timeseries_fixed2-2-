import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["CUDA_VISIBLE_DEVICES"]  = "-1"

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

TF_AVAILABLE = False
try:
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    tf.autograph.set_verbosity(0)
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, Callback
    TF_AVAILABLE = True
except Exception as e:
    print(f"[LSTM] TensorFlow unavailable: {e}")

if TF_AVAILABLE:
    class ProgressCallback(Callback):
        def __init__(self, total_epochs, cb):
            super().__init__()
            self.total = total_epochs
            self.cb    = cb
        def on_epoch_end(self, epoch, logs=None):
            pct  = 20 + int(70 * (epoch + 1) / self.total)
            loss = logs.get("loss", 0)
            val  = logs.get("val_loss", 0)
            msg  = f"Epoch {epoch+1}/{self.total} — loss={loss:.5f}  val={val:.5f}"
            print(f"  {msg}")
            if self.cb:
                self.cb(msg, pct)


def generate_sample_data(dataset_type: str, n_points: int = 300) -> dict:
    np.random.seed(42)
    x = np.arange(n_points)
    if dataset_type == "stock":
        values = 100 + x*0.3 + 20*np.sin(x*0.1) + 10*np.sin(x*0.05) + np.random.normal(0, 5, n_points)
        label  = "Stock Price ($)"
    elif dataset_type == "sine":
        values = 50 * np.sin(x * 0.1) + 50 + np.random.normal(0, 2, n_points)
        label  = "Amplitude"
    elif dataset_type == "seasonal":
        values = 20 + 30*np.sin(x*2*np.pi/52) + x*0.05 + np.random.normal(0, 3, n_points)
        label  = "Temperature (°C)"
    else:
        values = np.cumsum(np.random.normal(0, 1, n_points)) + 100
        label  = "Value"
    dates = pd.date_range(start="2020-01-01", periods=n_points, freq="D")
    return {
        "dates":  [d.strftime("%Y-%m-%d") for d in dates],
        "values": values.tolist(),
        "label":  label,
    }


def preprocess_data(values, lookback=30):
    data   = np.array(values, dtype=np.float32).reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data)
    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback : i, 0])
        y.append(scaled[i, 0])
    return (np.array(X, dtype=np.float32).reshape(-1, lookback, 1),
            np.array(y, dtype=np.float32),
            scaler)


def train_and_predict(values, lookback=30, epochs=30, units=64,
                      dropout=0.2, future_steps=30, progress_cb=None):
    def cb(msg, pct=None):
        if progress_cb: progress_cb(msg, pct)

    if not TF_AVAILABLE:
        cb("TensorFlow not found — using moving average fallback", 20)
        return _moving_average_fallback(values, lookback, future_steps, progress_cb)

    cb("Scaling data to [0,1]...", 15)
    X, y, scaler = preprocess_data(values, lookback)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    cb(f"Building LSTM model ({units} units)...", 18)
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(dropout),
        LSTM(max(units // 2, 8)),
        Dropout(dropout),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")

    cb("Training started...", 20)
    history = model.fit(
        X_train, y_train,
        epochs           = epochs,
        batch_size       = 32,
        validation_split = 0.1,
        callbacks        = [
            EarlyStopping(monitor="val_loss", patience=7,
                          restore_best_weights=True, verbose=0),
            ProgressCallback(epochs, progress_cb),
        ],
        verbose = 0,
    )

    cb("Running test predictions...", 91)
    test_pred   = scaler.inverse_transform(
        model.predict(X_test, verbose=0)).flatten().tolist()
    test_actual = scaler.inverse_transform(
        y_test.reshape(-1, 1)).flatten().tolist()

    cb("Generating future forecast...", 93)
    last_seq = X[-1].copy()
    future_preds = []
    for _ in range(future_steps):
        p = float(model.predict(last_seq.reshape(1, lookback, 1), verbose=0)[0, 0])
        future_preds.append(float(scaler.inverse_transform([[p]])[0, 0]))
        last_seq = np.roll(last_seq, -1, axis=0)
        last_seq[-1, 0] = p

    cb("Computing metrics...", 96)
    n    = min(len(test_actual), len(test_pred))
    mae  = mean_absolute_error(test_actual[:n], test_pred[:n])
    rmse = float(np.sqrt(mean_squared_error(test_actual[:n], test_pred[:n])))
    mape = float(np.mean(np.abs(
        (np.array(test_actual[:n]) - np.array(test_pred[:n]))
        / (np.abs(np.array(test_actual[:n])) + 1e-8)
    )) * 100)

    return {
        "success":          True,
        "mode":             "LSTM Neural Network",
        "test_actual":      test_actual,
        "test_predicted":   test_pred,
        "future_predicted": future_preds,
        "train_loss":       [float(v) for v in history.history["loss"]],
        "val_loss":         [float(v) for v in history.history.get("val_loss", [])],
        "metrics":          {"mae": round(mae,4), "rmse": round(rmse,4), "mape": round(mape,2)},
        "split_index":      lookback + split,
        "lookback":         lookback,
        "epochs_ran":       len(history.history["loss"]),
    }


def _moving_average_fallback(values, lookback, future_steps, progress_cb=None):
    def cb(msg, pct=None):
        if progress_cb: progress_cb(msg, pct)
    cb("Computing moving averages...", 40)
    arr   = np.array(values, dtype=float)
    preds = [float(np.mean(arr[i-lookback:i])) for i in range(lookback, len(arr))]
    split = int(len(preds) * 0.8)
    test_actual = arr[lookback+split:].tolist()
    test_pred   = preds[split:]
    n           = min(len(test_actual), len(test_pred))
    cb("Generating future predictions...", 80)
    last_window  = arr[-lookback:].tolist()
    future_preds = []
    for _ in range(future_steps):
        nxt = float(np.mean(last_window))
        future_preds.append(nxt)
        last_window = last_window[1:] + [nxt]
    mae  = mean_absolute_error(test_actual[:n], test_pred[:n])
    rmse = float(np.sqrt(mean_squared_error(test_actual[:n], test_pred[:n])))
    return {
        "success": True,
        "mode":    "Moving Average (TensorFlow not installed)",
        "test_actual": test_actual, "test_predicted": test_pred[:n],
        "future_predicted": future_preds,
        "train_loss": [], "val_loss": [],
        "metrics": {"mae": round(mae,4), "rmse": round(rmse,4), "mape": 0.0},
        "split_index": lookback+split, "lookback": lookback, "epochs_ran": 0,
    }