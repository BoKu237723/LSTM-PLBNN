import numpy as np
import pandas as pd
from pathlib import Path
import time
import json
import threading
import os
import psutil

np.random.seed(42)

# Activation Functions
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def dsigmoid(x):
    return x * (1.0 - x)

def dtanh(x):
    return 1.0 - x ** 2

# NumPy LSTM
class NumpyLSTM:
    def __init__(self, input_size=4, hidden_size=32, output_size=1, learning_rate=0.005):

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = learning_rate

        limit = np.sqrt(6 / (hidden_size + input_size + hidden_size))

        # Gates
        self.Wf = np.random.uniform(-limit, limit, (hidden_size, hidden_size + input_size))
        self.Wi = np.random.uniform(-limit, limit, (hidden_size, hidden_size + input_size))
        self.Wc = np.random.uniform(-limit, limit, (hidden_size, hidden_size + input_size))
        self.Wo = np.random.uniform(-limit, limit, (hidden_size, hidden_size + input_size))

        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))

        # Output layer
        self.Wy = np.random.uniform(-limit, limit,(output_size, hidden_size))
        self.by = np.zeros((output_size, 1))

    # Forward
    def forward(self, x_sequence):

        h = np.zeros((self.hidden_size, 1))
        c = np.zeros((self.hidden_size, 1))

        caches = []

        for x in x_sequence:

            x = x.reshape(-1, 1)

            concat = np.vstack((h, x)) # Input Concatenation >> zt​=[ht−1​,xt​]

            f = sigmoid(self.Wf.dot(concat) + self.bf) # Forget Gate >>  ft ​= σ(Wf​zt​+bf​)
            i = sigmoid(self.Wi.dot(concat) + self.bi) # Input Gate >> i = sigmoid(self.Wi.dot(concat) + self.bi)

            c_bar = np.tanh(self.Wc.dot(concat) + self.bc) # Candidate Celll State >> (C^~ t ​= tanh(Wc​zt​+bc​))

            c = f * c + i * c_bar # Cell State Update >> Ct ​= ft​⊙Ct−1​+it​⊙C~t​

            o = sigmoid(self.Wo.dot(concat) + self.bo) # Output Gate >> ot ​= σ(Wo​zt​+bo​)

            h = o * np.tanh(c) # Hidden State >> ht ​= ot​⊙tanh(Ct​)

            caches.append((x, h, c, c_bar, f, i, o, concat))

        y = self.Wy.dot(h) + self.by # Final Output Layer >> y^ ​= Wy​ht​+by​

        return y, caches

    # Backward Pass
    def backward(self, y_pred, y_true, caches):

        grads = {
            k: np.zeros_like(v)
            for k, v in self.__dict__.items()
            if k.startswith(("W", "b"))
        }

        dy = y_pred - y_true # Gradient of the Output >> ∂L/∂y^​^ = y^^​−y

        grads["Wy"] = dy.dot(caches[-1][1].T)
        grads["by"] = dy

        dh_next = self.Wy.T.dot(dy)
        dc_next = np.zeros((self.hidden_size, 1))

        for t in reversed(range(len(caches))): # Backpropagation Through Time (BPTT) >> ∂L​/∂W = ∑^T ​∂L/​∂ht ∂ht​​/​∂W
            #                                                                                  t=1

            x, h, c, c_bar, f, i, o, concat = caches[t]

            c_prev = caches[t - 1][2] if t > 0 else np.zeros_like(c)

            do_raw = (dh_next * np.tanh(c)) * dsigmoid(o)

            dc = (dh_next * o * (1 - np.tanh(c) ** 2)) + dc_next

            di_raw = (dc * c_bar) * dsigmoid(i)

            dc_bar_raw = (dc * i) * dtanh(c_bar)

            df_raw = (dc * c_prev) * dsigmoid(f)

            grads["Wf"] += df_raw.dot(concat.T)
            grads["Wi"] += di_raw.dot(concat.T)
            grads["Wc"] += dc_bar_raw.dot(concat.T)
            grads["Wo"] += do_raw.dot(concat.T)

            grads["bf"] += df_raw
            grads["bi"] += di_raw
            grads["bc"] += dc_bar_raw
            grads["bo"] += do_raw

            dconcat = (
                self.Wf.T.dot(df_raw)
                + self.Wi.T.dot(di_raw)
                + self.Wc.T.dot(dc_bar_raw)
                + self.Wo.T.dot(do_raw)
            )

            dh_next = dconcat[:self.hidden_size, :]

            dc_next = dc * f

        return grads

    # Training
    def train(self, X, y, epochs=50, print_every=10):

        for epoch in range(1, epochs + 1):

            epoch_loss = 0.0

            for x_seq, target in zip(X, y):

                y_pred, caches = self.forward(x_seq)

                loss = 0.5 * np.sum((y_pred - target) ** 2) # Loss Function (Mean Squared Err) >> L = 1/2 ​(y−y^^​)^2 # For multiple outptut >> L = 1/2 ​∑i ​(y^^​i​−yi​)^2

                epoch_loss += loss

                grads = self.backward(y_pred, target, caches)

                # Gradient clipping
                for key in grads:

                    grad = grads[key]

                    np.clip(grad, -1.0, 1.0, out=grad)

                    setattr(
                        self,
                        key,
                        getattr(self, key) - self.lr * grad # θ=θ−η∇θ​L
                    )

            if epoch % print_every == 0 or epoch == 1:

                print(
                    f"Epoch {epoch}/{epochs} | "
                    f"Loss = {epoch_loss / len(X):.6f}"
                )

# Data Preparation
def load_and_prepare_data(csv_path, seq_len=48):

    df = pd.read_csv(csv_path, parse_dates=["timestamp"])

    # Cyclical Time Features 
    hours = (
        df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    ) # 10:30 -> 10.5

    df["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)

    # Weekend Feature
    df["is_weekend"] = (
        df["timestamp"].dt.weekday >= 5
    ).astype(float)

    # Normalize Traffic
    values = df["traffic_mb"].values.astype(float)

    min_val = values.min()
    max_val = values.max()

    df["traffic_norm"] = (
        (values - min_val) / (max_val - min_val + 1e-8)
    )

    feature_cols = [
        "traffic_norm",
        "hour_sin",
        "hour_cos",
        "is_weekend"
    ]

    data_matrix = df[feature_cols].values

    X = []
    y = []

    for i in range(len(data_matrix) - seq_len):

        X.append(data_matrix[i:i + seq_len])

        y.append(data_matrix[i + seq_len, 0])

    return (
        np.array(X),                    # training sequences
        np.array(y).reshape(-1, 1),     # targets, aka ground truth
        min_val,                        # min traffic
        max_val,                        # max traffix
        df                              # original dataframe
    )

def main():

    csv_path = Path("network_traffic_data.csv")

    if not csv_path.exists():

        print("ERR: network_traffic_data.csv not found.")
        return

    seq_len = 48 # 1 per 30 min

    X_all, y_all, min_val, max_val, df_orig = (
        load_and_prepare_data(csv_path, seq_len)
    )

    train_raw_steps = 3 * 7 * 48
    train_size = train_raw_steps - seq_len

    X_train = X_all[:train_size]
    y_train = y_all[:train_size]

    model = NumpyLSTM(
        input_size=4,
        hidden_size=32,
        learning_rate=0.005 # new rate/ prv:0.0001
    )
    print("Starting Training...\n")

    # Start memory monitor
    def start_mem_monitor():
        m = {"max_rss": 0}
        stop = threading.Event()

        def _run():
            pid = os.getpid()
            proc = psutil.Process(pid) if psutil else None
            while not stop.is_set():
                if proc:
                    try:
                        rss = proc.memory_info().rss
                        if rss > m["max_rss"]:
                            m["max_rss"] = rss
                    except Exception:
                        pass
                time.sleep(0.01)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return t, stop, m

    def estimate_flops(input_size, hidden_size, output_size, seq_len, dataset_size, epochs, future_steps, backward_factor=3):
        per_timestep_mac = 4 * hidden_size * (hidden_size + input_size)
        wy_mac = output_size * hidden_size
        forward_mac_per_sample = seq_len * (per_timestep_mac + wy_mac)
        training_mac_per_sample = forward_mac_per_sample * (1 + backward_factor)
        total_training_mac = training_mac_per_sample * dataset_size * epochs
        inference_mac = forward_mac_per_sample * future_steps
        total_mac = total_training_mac + inference_mac
        flops = int(total_mac * 2)  # assume 2 FLOPs per MAC

        return flops

    monitor_thread, monitor_stop, monitor_state = start_mem_monitor()
    t_start = time.perf_counter()

    model.train(
        X_train,
        y_train,
        epochs=50,
        print_every=10
    )


    # Forecast using REAL HISTORY (not the past prediction history)
    print("\nGenerating Predictions...\n")

    future_steps = len(X_all) - train_size

    predictions_norm = []

    test_start = train_size

    for i in range(future_steps):

        current_sequence = X_all[test_start + i]

        y_pred, _ = model.forward(current_sequence)

        # y_pred.shape

        pred_val = y_pred.item()

        # Some noise
        noise = np.random.normal(0, 0.01)

        pred_val += noise

        # Clamp normalized range
        pred_val = max(0.0, min(1.0, pred_val))

        predictions_norm.append(pred_val)

    # Denormalize
    predictions = (
        np.array(predictions_norm) * (max_val - min_val) + min_val
    )

    # Future Timestamps
    future_timestamps = df_orig["timestamp"].iloc[
        train_size + seq_len :
        train_size + seq_len + future_steps
    ].reset_index(drop=True)

    actual = df_orig["traffic_mb"].iloc[
        train_size + seq_len :
        train_size + seq_len + future_steps
    ].values
    
    predicted = predictions
    
    # RMSE
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    
    # MAE
    mae = np.mean(np.abs(actual - predicted))
    
    # MAPE (%)
    mask = actual != 0  # avoid divide-by-zero
    mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    
    print("\n========== Model Evaluation ==========")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"MAPE : {mape:.2f}%")

    # Save Forecast
    forecast_df = pd.DataFrame({
        "timestamp": future_timestamps,
        "predicted_traffic_mb": predictions
    })

    forecast_path = Path(
        "network_traffic_forecast.csv"
    )

    forecast_df.to_csv(forecast_path, index=False)

    print(f"Forecast saved to: {forecast_path}")

    # Stop monitor and record results
    t_end = time.perf_counter()
    monitor_stop.set()
    monitor_thread.join(timeout=1.0)

    duration = float(t_end - t_start)
    peak_mem = int(monitor_state.get("max_rss", 0)) if monitor_state else None

    flops = estimate_flops(
        input_size=4,
        hidden_size=32,
        output_size=1,
        seq_len=seq_len,
        dataset_size=len(X_train),
        epochs=50,
        future_steps=future_steps,
    )

    results = {
        "model": "lstm_from_scratch",
        "duration_seconds": duration,
        "peak_memory_bytes": peak_mem,
        "estimated_flops": flops,
        "RMSE" : rmse,
        "MAE": mae,
        "MAPE": mape,
    }

    try:
        out_path = Path("benchmark_results_lstm_from_scratch.json")

        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)

        print(f"Benchmark results saved to: {out_path}")

    except Exception as e:
        print(f"Failed to save benchmark results: {e}")

if __name__ == "__main__":
    main()