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

def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

# Bayesian NumPy LSTM with Gaussian variational posterior
class NumpyLSTM:
    def __init__(self, input_size=4, hidden_size=32, output_size=1, learning_rate=0.005, prior_sigma=1.0, likelihood_sigma=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = learning_rate
        self.prior_sigma = prior_sigma
        self.likelihood_sigma = likelihood_sigma

        limit = np.sqrt(6 / (hidden_size + input_size + hidden_size))

        self.param_names = [
            "Wf", "Wi", "Wc", "Wo", "Wy",
            "bf", "bi", "bc", "bo", "by"
        ]

        # Dynamically set _mu and _rho attributes on the class object
        for name in self.param_names:
            if name.startswith("W"):
                if name == "Wy":
                    mu_val = np.random.uniform(-limit, limit, (output_size, hidden_size))
                else:
                    mu_val = np.random.uniform(-limit, limit, (hidden_size, hidden_size + input_size))
            else: # Biases
                if name == "by":
                    mu_val = np.zeros((output_size, 1))
                else:
                    mu_val = np.zeros((hidden_size, 1))
            
            rho_val = np.full_like(mu_val, -5.0)
            
            # This mimics the standard LSTM layout dynamically
            setattr(self, f"{name}_mu", mu_val)
            setattr(self, f"{name}_rho", rho_val)

        self.sampled = {}

    def sample_weights(self):
        self.sampled = {}

        for name in self.param_names:
            mu = getattr(self, f"{name}_mu")
            rho = getattr(self, f"{name}_rho")
            sigma = softplus(rho)
            eps = np.random.randn(*mu.shape)
            value = mu + sigma * eps
            self.sampled[name] = {
                "value": value,
                "sigma": sigma,
                "eps": eps,
            }
        return self.sampled
    
    def sample(self, name):
        return self.sampled[name]["value"]

    def kl_divergence(self):
        kl = 0.0
        for name in self.param_names:
            mu = getattr(self, f"{name}_mu")
            rho = getattr(self, f"{name}_rho")
            sigma = softplus(rho)
            kl += np.sum(
                np.log(self.prior_sigma / sigma)
                + (sigma ** 2 + mu ** 2) / (2.0 * self.prior_sigma ** 2) - 0.5
            )
        return kl

    def forward(self, x_sequence):
        self.sample_weights()

        h = np.zeros((self.hidden_size, 1))
        c = np.zeros((self.hidden_size, 1))
        caches = []

        for x in x_sequence:
            x = x.reshape(-1, 1)
            concat = np.vstack((h, x))

            f = sigmoid(self.sample("Wf").dot(concat) + self.sample("bf"))
            i = sigmoid(self.sample("Wi").dot(concat) + self.sample("bi"))
            c_bar = np.tanh(self.sample("Wc").dot(concat) + self.sample("bc"))
            c = f * c + i * c_bar
            o = sigmoid(self.sample("Wo").dot(concat) + self.sample("bo"))
            h = o * np.tanh(c)

            caches.append((x, h, c, c_bar, f, i, o, concat))

        y = self.sample("Wy").dot(h) + self.sample("by")
        return y, caches

    def backward(self, y_pred, y_true, caches, kl_scale):
        grads_w = {
            name: np.zeros_like(getattr(self, f"{name}_mu"))
            for name in self.param_names
        }

        dy = (y_pred - y_true) / (self.likelihood_sigma ** 2)

        grads_w["Wy"] = dy.dot(caches[-1][1].T)
        grads_w["by"] = dy

        dh_next = self.sample("Wy").T.dot(dy)
        dc_next = np.zeros((self.hidden_size, 1))

        for t in reversed(range(len(caches))):
            x, h, c, c_bar, f, i, o, concat = caches[t]
            c_prev = caches[t - 1][2] if t > 0 else np.zeros_like(c)

            do_raw = (dh_next * np.tanh(c)) * dsigmoid(o)
            dc = (dh_next * o * (1 - np.tanh(c) ** 2)) + dc_next
            di_raw = (dc * c_bar) * dsigmoid(i)
            dc_bar_raw = (dc * i) * dtanh(c_bar)
            df_raw = (dc * c_prev) * dsigmoid(f)

            grads_w["Wf"] += df_raw.dot(concat.T)
            grads_w["Wi"] += di_raw.dot(concat.T)
            grads_w["Wc"] += dc_bar_raw.dot(concat.T)
            grads_w["Wo"] += do_raw.dot(concat.T)
            grads_w["bf"] += df_raw
            grads_w["bi"] += di_raw
            grads_w["bc"] += dc_bar_raw
            grads_w["bo"] += do_raw

            dconcat = (
                self.sample("Wf").T.dot(df_raw)
                + self.sample("Wi").T.dot(di_raw)
                + self.sample("Wc").T.dot(dc_bar_raw)
                + self.sample("Wo").T.dot(do_raw)
            )

            dh_next = dconcat[: self.hidden_size, :]
            dc_next = dc * f

        grads = {}
        for name in self.param_names:
            mu = getattr(self, f"{name}_mu")
            rho = getattr(self, f"{name}_rho")
            sigma = softplus(rho)
            eps = self.sampled[name]["eps"]

            dmu = grads_w[name]
            dsigma = grads_w[name] * eps
            drho = dsigma * sigmoid(rho)

            dmu += kl_scale * (mu / (self.prior_sigma ** 2))
            drho += kl_scale * (-1.0 / sigma + sigma / (self.prior_sigma ** 2)) * sigmoid(rho)

            np.clip(dmu, -1.0, 1.0, out=dmu)
            np.clip(drho, -1.0, 1.0, out=drho)

            # Store matching names to iterate through later
            grads[name + "_mu"] = dmu
            grads[name + "_rho"] = drho

        return grads

    # Training
    def train(self, X, y, epochs=50, print_every=10):
        dataset_size = len(X)

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            epoch_kl = 0.0

            for x_seq, target in zip(X, y):
                y_pred, caches = self.forward(x_seq)

                mse_loss = 0.5 * np.sum((y_pred - target) ** 2) / (self.likelihood_sigma ** 2)
                kl = self.kl_divergence()
                loss = mse_loss + kl / dataset_size

                epoch_loss += loss
                epoch_kl += kl / dataset_size

                grads = self.backward(y_pred, target, caches, kl_scale=1.0 / dataset_size)

                # DYNAMIC LOOKUP AND UPDATE MATCHING THE STANDARD LSTM
                for key in grads:
                    grad = grads[key]
                    # Direct weight update syntax via reflection strings
                    setattr(
                        self,
                        key,
                        getattr(self, key) - self.lr * grad
                    )

            if epoch % print_every == 0 or epoch == 1:
                print(
                    f"Epoch {epoch}/{epochs} | "
                    f"Loss = {epoch_loss / dataset_size:.6f} | "
                    f"KL = {epoch_kl / dataset_size:.6f}"
                )

    def predict_with_uncertainty(self, x_sequence, samples=20):
        preds = []
        for _ in range(samples):

            y_pred, _ = self.forward(x_sequence)
            preds.append(y_pred.item())

        preds = np.array(preds)
        return preds.mean(), preds.std(), preds

# Data Preparation
def load_and_prepare_data(csv_path, seq_len=48):

    df = pd.read_csv(csv_path, parse_dates=["timestamp"])

    # Cyclical Time Features 
    hours = (
        df["timestamp"].dt.hour
        + df["timestamp"].dt.minute / 60.0
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
        (values - min_val)
        / (max_val - min_val + 1e-8)
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

        print("ERROR: network_traffic_data.csv not found.")
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
        learning_rate=0.005, # 0.005
        prior_sigma=1.0,
        likelihood_sigma=0.2,
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

    def estimate_flops(input_size, hidden_size, output_size, seq_len, dataset_size, epochs, future_steps, uncertainty_samples=20, backward_factor=3):
        gate_weights = 4 * hidden_size * (hidden_size + input_size)
        gate_bias = 4 * hidden_size
        output_weights = output_size * hidden_size
        output_bias = output_size

        num_params = gate_weights + gate_bias + output_weights + output_bias

        per_timestep_mac = 4 * hidden_size * (hidden_size + input_size)
        wy_mac = output_size * hidden_size
        forward_mac_per_sample = seq_len * (per_timestep_mac + wy_mac)

        # Bayesian overhead: sampling, KL, and posterior-gradient transforms
        sample_ops = int(num_params * 8)
        kl_ops = int(num_params * 8)
        backward_bayesian_ops = int(num_params * 4)

        bayesian_forward_flops = int(forward_mac_per_sample * 2) + sample_ops
        bayesian_training_flops_per_sample = bayesian_forward_flops + int(forward_mac_per_sample * 2) * backward_factor + backward_bayesian_ops + kl_ops
        total_training_flops = bayesian_training_flops_per_sample * dataset_size * epochs

        bayesian_inference_flops = (int(forward_mac_per_sample * 2) + sample_ops) * future_steps * uncertainty_samples
        total_flops = total_training_flops + bayesian_inference_flops

        return total_flops

    monitor_thread, monitor_stop, monitor_state = start_mem_monitor()
    t_start = time.perf_counter()

    model.train(
        X_train,
        y_train,
        epochs=50,
        print_every=10
    )

    # Forecast with uncertainty using REAL HISTORY
    print("\nGenerating Predictions with Uncertainty...\n")

    future_steps = len(X_all) - train_size
    predictions_norm = []
    predictions_std = []
    lower_bounds = []
    upper_bounds = []
    test_start = train_size

    for i in range(future_steps):
        current_sequence = X_all[test_start + i]
        mean_pred, std_pred, _ = model.predict_with_uncertainty(
            current_sequence,
            samples=20
        )

        mean_pred = max(0.0, min(1.0, mean_pred))
        lower = max(0.0, mean_pred - 2.0 * std_pred)
        upper = min(1.0, mean_pred + 2.0 * std_pred)

        predictions_norm.append(mean_pred)
        predictions_std.append(std_pred)
        lower_bounds.append(lower)
        upper_bounds.append(upper)

    predictions = (
        np.array(predictions_norm) * (max_val - min_val) + min_val
    )

    lower_predictions = (
        np.array(lower_bounds) * (max_val - min_val) + min_val
    )

    upper_predictions = (
        np.array(upper_bounds) * (max_val - min_val) + min_val
    )

    # Ground truth values (same timestamps as predictions)
    actual = df_orig["traffic_mb"].iloc[
        train_size + seq_len :
        train_size + seq_len + future_steps
    ].values


    # Average predictive uncertainty
    uncertainty_percentage = (
        np.mean(predictions_std) * (max_val - min_val)
        / np.mean(actual)
    ) * 100
    
    print(f"Average Uncertainty : {uncertainty_percentage:.2f}%")

    
    predicted = predictions
    
    # RMSE
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    
    # MAE
    mae = np.mean(np.abs(actual - predicted))
    
    # MAPE (%)
    mask = actual != 0  # divide-by-zero
    mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    
    print("\n========== Model Evaluation ==========")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"MAPE : {mape:.2f}%")

    future_timestamps = df_orig["timestamp"].iloc[
        train_size + seq_len :
        train_size + seq_len + future_steps
    ].reset_index(drop=True)

    forecast_df = pd.DataFrame({
        "timestamp": future_timestamps,
        "predicted_traffic_mb": predictions,
        "uncertainty_std_mb": np.array(predictions_std) * (max_val - min_val),
        "lower_bound_mb": lower_predictions,
        "upper_bound_mb": upper_predictions,
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
        "model": "lstm_bnn",
        "duration_seconds": duration,
        "peak_memory_bytes": peak_mem,
        "estimated_flops": flops,
        "RMSE" : rmse,
        "MAE": mae,
        "MAPE": mape,
        "average_uncertainty" : uncertainty_percentage
    }

    try:
        out_path = Path("benchmark_results_lstm_bnn.json")

        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)

        print(f"Benchmark results saved to: {out_path}")

    except Exception as e:
        print(f"Failed to save benchmark results: {e}")

if __name__ == "__main__":
    main()