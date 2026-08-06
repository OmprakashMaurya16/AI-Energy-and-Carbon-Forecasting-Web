import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

PROCESSED_PATH = os.path.join(_ROOT, "data", "processed", "merged_hourly.csv")
MODEL_PATH = os.path.join(_ROOT, "models", "xgboost_energy.json")
SCALER_PATH = os.path.join(_ROOT, "models", "scaler.pkl")
METRICS_PATH = os.path.join(_ROOT, "models", "metrics.json")

TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_processed() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_PATH, parse_dates=["datetime"], index_col="datetime")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    fe = df.copy()

    fe["hour"] = fe.index.hour
    fe["dayofweek"] = fe.index.dayofweek
    fe["month"] = fe.index.month
    fe["dayofyear"] = fe.index.dayofyear
    fe["is_weekend"] = (fe.index.dayofweek >= 5).astype(int)
    fe["hour_sin"] = np.sin(2 * np.pi * fe["hour"] / 24)
    fe["hour_cos"] = np.cos(2 * np.pi * fe["hour"] / 24)
    fe["month_sin"] = np.sin(2 * np.pi * fe["month"] / 12)
    fe["month_cos"] = np.cos(2 * np.pi * fe["month"] / 12)

    for lag in [1, 2, 3, 6, 12, 24, 48]:
        fe[f"lag_{lag}h"] = fe["energy_kwh"].shift(lag)

    fe["rolling_mean_3h"] = fe["energy_kwh"].shift(1).rolling(3).mean()
    fe["rolling_mean_24h"] = fe["energy_kwh"].shift(1).rolling(24).mean()
    fe["rolling_std_24h"] = fe["energy_kwh"].shift(1).rolling(24).std()
    fe["rolling_mean_168h"] = fe["energy_kwh"].shift(1).rolling(168).mean()

    fe = fe.dropna()
    return fe


def split_data(fe: pd.DataFrame):
    target = "energy_kwh"
    drop_cols = [target, "active_power_kw"]
    feature_cols = [c for c in fe.columns if c not in drop_cols]

    X = fe[feature_cols]
    y = fe[target]

    split_idx = int(len(fe) * (1 - TEST_SIZE))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(
        f"Train : {X_train.shape[0]} rows  ({X_train.index.min().date()} → {X_train.index.max().date()})"
    )
    print(
        f"Test  : {X_test.shape[0]} rows  ({X_test.index.min().date()} → {X_test.index.max().date()})"
    )
    print(f"Features : {X_train.shape[1]}")

    return X_train, X_test, y_train, y_test, feature_cols


def train_model(X_train, y_train):
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        early_stopping_rounds=30,
        eval_metric="rmse",
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    split = int(len(X_train) * 0.9)
    X_tr, X_val = X_train_scaled[:split], X_train_scaled[split:]
    y_tr, y_val = y_train.iloc[:split], y_train.iloc[split:]

    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    print(f"\nBest iteration : {model.best_iteration}")
    return model, scaler


def evaluate_model(model, scaler, X_test, y_test):
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    y_pred = np.clip(y_pred, 0, None)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-9))) * 100

    metrics = {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
        "MAPE": round(mape, 4),
    }

    print("\n" + "=" * 45)
    print("  MODEL EVALUATION — Test Set")
    print("=" * 45)
    print(f"  MAE  : {mae:.4f} kWh")
    print(f"  RMSE : {rmse:.4f} kWh")
    print(f"  R²   : {r2:.4f}")
    print(f"  MAPE : {mape:.2f} %")
    print("=" * 45 + "\n")

    return metrics, y_pred


def save_artifacts(model, scaler, metrics):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save_model(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Model  saved → {MODEL_PATH}")
    print(f"Scaler saved → {SCALER_PATH}")
    print(f"Metrics saved → {METRICS_PATH}")


def load_model_artifacts():
    model = XGBRegressor()
    model.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    return model, scaler, metrics


def get_predictions_df(X_test, y_test, y_pred) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "actual": y_test.values,
            "predicted": y_pred,
        },
        index=y_test.index,
    )


def run_training_pipeline():
    print("Loading processed data …")
    df = load_processed()
    print(f"Shape : {df.shape}")

    print("\nBuilding features …")
    fe = build_features(df)
    print(f"Feature matrix shape : {fe.shape}")

    print("\nSplitting data …")
    X_train, X_test, y_train, y_test, feature_cols = split_data(fe)

    print("\nTraining XGBoost …")
    model, scaler = train_model(X_train, y_train)

    metrics, y_pred = evaluate_model(model, scaler, X_test, y_test)

    save_artifacts(model, scaler, metrics)

    pred_df = get_predictions_df(X_test, y_test, y_pred)
    return model, scaler, metrics, pred_df, feature_cols


if __name__ == "__main__":
    model, scaler, metrics, pred_df, feature_cols = run_training_pipeline()
    print("\nSample predictions vs actuals:")
    print(pred_df.head(10).to_string())
