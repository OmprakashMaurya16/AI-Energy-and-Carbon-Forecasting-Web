import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

PROCESSED_PATH = os.path.join(_ROOT, "data", "processed", "merged_hourly.csv")
MODEL_PATH = os.path.join(_ROOT, "models", "lstm_energy.pt")
FEATURE_SCALER_PATH = os.path.join(_ROOT, "models", "feature_scaler.pkl")
TARGET_SCALER_PATH = os.path.join(_ROOT, "models", "target_scaler.pkl")
CONFIG_PATH = os.path.join(_ROOT, "models", "model_config.json")
METRICS_PATH = os.path.join(_ROOT, "models", "metrics.json")

TARGET_COL = "energy_kwh"
DROP_COLS = ["active_power_kw"]  # exact duplicate of energy_kwh in this dataset

# --- Sequence framing ---------------------------------------------------
LOOKBACK_HOURS = 168   # 1 week of history feeds the model
FORECAST_HORIZON = 24  # predict the next 24 hours in one shot

# --- Chronological split (NOT random — this is a time series) ----------
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# remaining 0.15 is test

# --- Model / training hyperparameters -----------------------------------
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
MAX_EPOCHS = 100
EARLY_STOP_PATIENCE = 10
RANDOM_STATE = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================================
# 1. Load + lightweight calendar features
# =========================================================================
def load_processed() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_PATH, parse_dates=["datetime"], index_col="datetime")
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cyclical time-of-day / time-of-year features only.

    Unlike the XGBoost version, we deliberately do NOT add lag_*/rolling_*
    columns here — the LSTM already sees the last LOOKBACK_HOURS of every
    raw feature at every step, so those engineered lags would just be
    redundant, differently-shaped copies of information the recurrent
    layer already has access to.
    """
    fe = df.copy()
    hour = fe.index.hour
    month = fe.index.month

    fe["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    fe["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    fe["month_sin"] = np.sin(2 * np.pi * month / 12)
    fe["month_cos"] = np.cos(2 * np.pi * month / 12)
    fe["is_weekend"] = (fe.index.dayofweek >= 5).astype(int)

    fe = fe.drop(columns=[c for c in DROP_COLS if c in fe.columns])
    return fe


# =========================================================================
# 2. Chronological split -> fit scalers on TRAIN ONLY -> scale everything
# =========================================================================
def chronological_split_indices(n_rows: int):
    train_end = int(n_rows * TRAIN_FRAC)
    val_end = int(n_rows * (TRAIN_FRAC + VAL_FRAC))
    return train_end, val_end


def fit_and_scale(fe: pd.DataFrame):
    """
    Fit a feature scaler and a target scaler using ONLY the train-range
    rows, then apply both across the full continuous series. Windows are
    built afterwards (see build_windows) — this ordering is what keeps
    val/test performance honest.
    """
    feature_cols = [c for c in fe.columns if c != TARGET_COL]
    train_end, val_end = chronological_split_indices(len(fe))

    feature_scaler = StandardScaler()
    feature_scaler.fit(fe[feature_cols].iloc[:train_end])

    target_scaler = StandardScaler()
    target_scaler.fit(fe[[TARGET_COL]].iloc[:train_end])

    scaled = fe.copy()
    scaled[feature_cols] = feature_scaler.transform(fe[feature_cols])
    scaled[TARGET_COL] = target_scaler.transform(fe[[TARGET_COL]])

    return scaled, feature_cols, feature_scaler, target_scaler, train_end, val_end


# =========================================================================
# 3. Sliding windows: X = [t-168 .. t-1] all features, y = [t .. t+23] energy
# =========================================================================
def build_windows(scaled: pd.DataFrame, feature_cols: list, train_end: int, val_end: int):
    """
    Build every valid (lookback -> horizon) window over the WHOLE
    continuous series, then assign each window to train/val/test by
    where its FORECAST (label) region starts. A val/test window's input
    may reach back into train-period rows — that's realistic (a real
    forecaster has access to real history) — but no val/test label was
    ever seen by the scaler fit above.
    """
    values_X = scaled[feature_cols + [TARGET_COL]].values  # target is also an input feature (autoregressive)
    values_y = scaled[TARGET_COL].values

    n = len(scaled)
    last_start = n - LOOKBACK_HOURS - FORECAST_HORIZON + 1

    X_list, y_list, split_tag = [], [], []
    for start in range(last_start):
        label_start = start + LOOKBACK_HOURS
        X_list.append(values_X[start:label_start])
        y_list.append(values_y[label_start:label_start + FORECAST_HORIZON])

        if label_start < train_end:
            split_tag.append("train")
        elif label_start < val_end:
            split_tag.append("val")
        else:
            split_tag.append("test")

    X = np.stack(X_list).astype(np.float32)
    y = np.stack(y_list).astype(np.float32)
    split_tag = np.array(split_tag)

    print(f"      Total windows        : {len(X)}")
    for tag in ["train", "val", "test"]:
        print(f"      {tag:5s} windows      : {(split_tag == tag).sum()}")

    return (
        X[split_tag == "train"], y[split_tag == "train"],
        X[split_tag == "val"], y[split_tag == "val"],
        X[split_tag == "test"], y[split_tag == "test"],
    )


class EnergySequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =========================================================================
# 4. Model
# =========================================================================
class LSTMForecaster(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, num_layers: int,
                 horizon: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        # x: (batch, lookback, n_features)
        out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]          # (batch, hidden_size) — final layer's last timestep
        return self.head(last_hidden)  # (batch, horizon)


# =========================================================================
# 5. Training loop with early stopping
# =========================================================================
def train_model(X_train, y_train, X_val, y_val, n_features: int):
    torch.manual_seed(RANDOM_STATE)

    train_loader = DataLoader(
        EnergySequenceDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        EnergySequenceDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False
    )

    model = LSTMForecaster(
        n_features=n_features,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        horizon=FORECAST_HORIZON,
        dropout=DROPOUT,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    GRAD_CLIP_NORM = 1.0

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    print(f"\nTraining on device: {DEVICE}")
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_losses = []
        for batch_idx, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            if epoch == 1 and batch_idx == 0:
                # One-time diagnostic: if this ever fires, the problem is
                # bad data (NaN/Inf already in the window), NOT the model —
                # go back and check data_engineering.py's cleaning, not this file.
                assert torch.isfinite(xb).all(), "Non-finite value in a training input window — data problem, not a model problem."
                assert torch.isfinite(yb).all(), "Non-finite value in a training label window — data problem, not a model problem."

            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            # LSTMs are prone to exploding gradients early in training;
            # without clipping, one bad batch can push weights to NaN and
            # every epoch after that reports NaN loss regardless of data quality.
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                val_losses.append(loss_fn(pred, yb).item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        print(f"  Epoch {epoch:3d} | train MSE {train_loss:.4f} | val MSE {val_loss:.4f}")

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"  Early stopping — no val improvement in {EARLY_STOP_PATIENCE} epochs.")
                break

    if best_state is None:
        raise RuntimeError(
            "Training completed without ever improving val loss (best_state is None) — "
            "this means every epoch's val loss was NaN or inf. If the diagnostic assertion "
            "above didn't fire, re-check for extreme outliers in energy_kwh/current_a in "
            "the raw data (only voltage/frequency are currently outlier-clamped)."
        )
    model.load_state_dict(best_state)
    print(f"\nBest val MSE : {best_val_loss:.4f}")
    return model


# =========================================================================
# 6. Evaluation (inverse-scaled, so metrics are in real kWh)
# =========================================================================
def evaluate_model(model, X_test, y_test, target_scaler):
    model.eval()
    test_loader = DataLoader(EnergySequenceDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    preds_scaled = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(DEVICE)
            preds_scaled.append(model(xb).cpu().numpy())
    preds_scaled = np.concatenate(preds_scaled, axis=0)  # (n_windows, horizon)

    # Inverse-transform: scaler was fit on shape (n, 1), so reshape through it per horizon step
    def inverse(arr_scaled):
        flat = arr_scaled.reshape(-1, 1)
        return target_scaler.inverse_transform(flat).reshape(arr_scaled.shape)

    y_pred = np.clip(inverse(preds_scaled), 0, None)
    y_true = inverse(y_test)

    mae = mean_absolute_error(y_true.ravel(), y_pred.ravel())
    rmse = np.sqrt(mean_squared_error(y_true.ravel(), y_pred.ravel()))
    r2 = r2_score(y_true.ravel(), y_pred.ravel())
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100

    # Per-horizon-step breakdown (useful for showing "predictions degrade further out")
    per_step_mae = [
        mean_absolute_error(y_true[:, h], y_pred[:, h]) for h in range(FORECAST_HORIZON)
    ]

    metrics = {
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "R2": round(float(r2), 4),
        "MAPE": round(float(mape), 4),
        "per_horizon_step_MAE": [round(float(m), 4) for m in per_step_mae],
    }

    print("\n" + "=" * 45)
    print("  MODEL EVALUATION — Test Set (24h-ahead)")
    print("=" * 45)
    print(f"  MAE  : {mae:.4f} kWh")
    print(f"  RMSE : {rmse:.4f} kWh")
    print(f"  R²   : {r2:.4f}")
    print(f"  MAPE : {mape:.2f} %")
    print(f"  MAE at hour+1 vs hour+24 : {per_step_mae[0]:.4f} → {per_step_mae[-1]:.4f}")
    print("=" * 45 + "\n")

    return metrics, y_pred, y_true


# =========================================================================
# 7. Persistence
# =========================================================================
def save_artifacts(model, feature_scaler, target_scaler, feature_cols, metrics):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    joblib.dump(feature_scaler, FEATURE_SCALER_PATH)
    joblib.dump(target_scaler, TARGET_SCALER_PATH)

    config = {
        "n_features": len(feature_cols) + 1,  # +1 because target is also fed as an input
        "feature_cols": feature_cols,
        "target_col": TARGET_COL,
        "lookback_hours": LOOKBACK_HOURS,
        "forecast_horizon": FORECAST_HORIZON,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Model    saved → {MODEL_PATH}")
    print(f"Scalers  saved → {FEATURE_SCALER_PATH}, {TARGET_SCALER_PATH}")
    print(f"Config   saved → {CONFIG_PATH}")
    print(f"Metrics  saved → {METRICS_PATH}")


def load_model_artifacts():
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    model = LSTMForecaster(
        n_features=config["n_features"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        horizon=config["forecast_horizon"],
        dropout=config["dropout"],
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    feature_scaler = joblib.load(FEATURE_SCALER_PATH)
    target_scaler = joblib.load(TARGET_SCALER_PATH)
    with open(METRICS_PATH) as f:
        metrics = json.load(f)

    return model, feature_scaler, target_scaler, config, metrics


def get_predictions_df(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """
    y_true / y_pred are (n_windows, horizon). We keep only the hour+1
    (next-hour) column here for a simple actual-vs-predicted table —
    app.py can reach into the full arrays for the 24h fan-out chart.
    """
    return pd.DataFrame({
        "actual_next_hour": y_true[:, 0],
        "predicted_next_hour": y_pred[:, 0],
    })


# =========================================================================
# 8. Full pipeline
# =========================================================================
def run_training_pipeline():
    print("Loading processed data …")
    df = load_processed()
    print(f"Shape : {df.shape}")

    print("\nAdding calendar features …")
    fe = add_calendar_features(df)
    print(f"Feature matrix shape : {fe.shape}")

    print("\nFitting scalers on train range + scaling full series …")
    scaled, feature_cols, feature_scaler, target_scaler, train_end, val_end = fit_and_scale(fe)
    print(f"Features (excl. target) : {len(feature_cols)}")

    print("\nBuilding sliding windows "
          f"(lookback={LOOKBACK_HOURS}h, horizon={FORECAST_HORIZON}h) …")
    X_train, y_train, X_val, y_val, X_test, y_test = build_windows(
        scaled, feature_cols, train_end, val_end
    )

    n_features = X_train.shape[-1]
    print(f"\nTraining LSTM …")
    model = train_model(X_train, y_train, X_val, y_val, n_features)

    metrics, y_pred, y_true = evaluate_model(model, X_test, y_test, target_scaler)

    save_artifacts(model, feature_scaler, target_scaler, feature_cols, metrics)

    pred_df = get_predictions_df(y_true, y_pred)
    return model, feature_scaler, target_scaler, metrics, pred_df, feature_cols


if __name__ == "__main__":
    model, feature_scaler, target_scaler, metrics, pred_df, feature_cols = run_training_pipeline()
    print("\nSample predictions vs actuals (next-hour only):")
    print(pred_df.head(10).to_string())
    