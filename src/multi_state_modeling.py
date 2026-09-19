import json
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.multi_state_data import (
    STATE_EMISSION_FACTORS,
    STATE_REGIONS,
    download_or_load_posoco_data,
)

METRICS_OUT_PATH = os.path.join(_ROOT, "models", "multi_state_metrics.json")
PREDS_OUT_PATH = os.path.join(_ROOT, "models", "multi_state_predictions.json")

# Representative states across North, South, West, and East India
KEY_STATES = [
    "Maharashtra",
    "Gujarat",
    "Uttar Pradesh",
    "Tamil Nadu",
    "Karnataka",
    "Delhi",
    "Rajasthan",
    "West Bengal",
    "Punjab",
    "Kerala",
]

LOOKBACK_DAYS = 14
FORECAST_HORIZON = 7
DEVICE = torch.device("cpu")


class StateLSTM(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, FORECAST_HORIZON)

    def forward(self, x):
        out, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])


def build_state_windows(series: np.ndarray, lookback=LOOKBACK_DAYS, horizon=FORECAST_HORIZON):
    X, y = [], []
    for i in range(len(series) - lookback - horizon + 1):
        X.append(series[i : i + lookback])
        y.append(series[i + lookback : i + lookback + horizon])
    return np.array(X)[..., np.newaxis], np.array(y)


def train_and_benchmark_states():
    """
    Trains sequence forecasting models on official POSOCO state-level data
    to validate model authenticity and generalizability across diverse Indian states.
    """
    os.makedirs(os.path.join(_ROOT, "models"), exist_ok=True)
    df = download_or_load_posoco_data()
    
    all_metrics = {}
    all_predictions = {}
    
    print("=" * 65)
    print("  MULTI-STATE FORECASTING BENCHMARK (POSOCO / GRID-INDIA)")
    print("=" * 65)
    
    for state in KEY_STATES:
        if state not in df.columns:
            continue
            
        series = df[state].values.astype(np.float32)
        dates = df.index
        
        # Train / Test split (80% train, 20% test)
        split_idx = int(len(series) * 0.80)
        train_vals = series[:split_idx]
        test_vals = series[split_idx:]
        
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_vals.reshape(-1, 1)).flatten()
        test_scaled = scaler.transform(test_vals.reshape(-1, 1)).flatten()
        
        X_train, y_train = build_state_windows(train_scaled)
        
        # Build test windows using lookback buffer from train
        full_scaled = np.concatenate([train_scaled[-LOOKBACK_DAYS:], test_scaled])
        X_test, y_test = build_state_windows(full_scaled)
        
        # PyTorch training
        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32)
        
        model = StateLSTM(hidden_dim=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()
        
        model.train()
        for epoch in range(40):
            optimizer.zero_grad()
            pred = model(X_t)
            loss = loss_fn(pred, y_t)
            loss.backward()
            optimizer.step()
            
        # Evaluation on unseen test set
        model.eval()
        with torch.no_grad():
            preds_scaled = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
            
        y_true_unscaled = scaler.inverse_transform(y_test)
        y_pred_unscaled = scaler.inverse_transform(preds_scaled)
        
        # Next-day evaluation metrics
        actual_1d = y_true_unscaled[:, 0]
        pred_1d = y_pred_unscaled[:, 0]
        
        mae = float(mean_absolute_error(actual_1d, pred_1d))
        rmse = float(np.sqrt(mean_squared_error(actual_1d, pred_1d)))
        r2 = float(r2_score(actual_1d, pred_1d))
        mape = float(np.mean(np.abs((actual_1d - pred_1d) / np.clip(actual_1d, 1e-3, None))) * 100)
        
        region = STATE_REGIONS.get(state, "Other")
        ef = STATE_EMISSION_FACTORS.get(state, 0.82)
        
        all_metrics[state] = {
            "region": region,
            "mean_demand_mu": round(float(np.mean(series)), 2),
            "mae_mu": round(mae, 2),
            "rmse_mu": round(rmse, 2),
            "r2": round(r2, 4),
            "mape_pct": round(mape, 2),
            "emission_factor": ef,
        }
        
        # Save dates and predictions for test window
        test_dates = [d.strftime("%Y-%m-%d") for d in dates[split_idx : split_idx + len(actual_1d)]]
        all_predictions[state] = {
            "dates": test_dates,
            "actual": [round(float(v), 2) for v in actual_1d],
            "predicted": [round(float(v), 2) for v in pred_1d],
        }
        
        print(f"  [{region:<8}] {state:<15} | R²: {r2:+.4f} | MAPE: {mape:5.2f}% | MAE: {mae:5.2f} MU")

    with open(METRICS_OUT_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)
        
    with open(PREDS_OUT_PATH, "w") as f:
        json.dump(all_predictions, f, indent=2)
        
    print("=" * 65)
    print(f"Saved benchmark metrics to {METRICS_OUT_PATH}")
    print(f"Saved benchmark predictions to {PREDS_OUT_PATH}")
    return all_metrics, all_predictions


def load_multi_state_artifacts():
    if not os.path.exists(METRICS_OUT_PATH) or not os.path.exists(PREDS_OUT_PATH):
        return train_and_benchmark_states()
    with open(METRICS_OUT_PATH) as f:
        metrics = json.load(f)
    with open(PREDS_OUT_PATH) as f:
        predictions = json.load(f)
    return metrics, predictions


if __name__ == "__main__":
    train_and_benchmark_states()
