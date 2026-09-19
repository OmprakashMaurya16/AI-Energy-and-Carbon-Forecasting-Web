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
import plotly.express as px
import plotly.graph_objects as go
import shap
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.modeling import (
    DEVICE,
    LOOKBACK_HOURS,
    FORECAST_HORIZON,
    add_calendar_features,
    build_windows,
    evaluate_model,
    fit_and_scale,
    get_predictions_df,
    load_model_artifacts,
    load_processed,
)

EMISSION_FACTOR_KG_PER_KWH = 0.82
METRICS_PATH = os.path.join(_ROOT, "models", "metrics.json")


def compute_shap_values(model, X_test: np.ndarray, feature_names: list = None, num_samples: int = None):
    """
    Compute Gradient x Input feature attributions for PyTorch LSTM sequence model
    and wrap them in a standard shap.Explanation object for visualization.
    """
    model.eval()
    if num_samples is not None:
        sample_len = min(num_samples, len(X_test))
        sample_X = X_test[:sample_len]
    else:
        sample_X = X_test

    x_tensor = torch.tensor(sample_X, dtype=torch.float32, requires_grad=True).to(DEVICE)
    preds = model(x_tensor)

    # Target: explain the first hour forecast (next-hour)
    target = preds[:, 0].sum()
    target.backward()

    grad = x_tensor.grad.detach().cpu().numpy()
    x_val = x_tensor.detach().cpu().numpy()

    # Time-averaged attribution per feature across the lookback window
    attributions = np.mean(grad * x_val, axis=1)
    data_summary = np.mean(x_val, axis=1)

    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(sample_X.shape[-1])]

    base_val = float(preds[:, 0].mean().detach().cpu().numpy())
    base_values = np.full((len(sample_X),), base_val)

    shap_values = shap.Explanation(
        values=attributions,
        base_values=base_values,
        data=data_summary,
        feature_names=feature_names,
    )
    return shap_values, data_summary


def plot_shap_bar(shap_values, top_n: int = 15) -> go.Figure:
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    feature_names = shap_values.feature_names

    importance_df = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": mean_abs_shap,
            }
        )
        .sort_values("importance", ascending=True)
        .tail(top_n)
    )

    fig = px.bar(
        importance_df,
        x="importance",
        y="feature",
        orientation="h",
        title=f"Feature Attribution (Top {top_n})",
        labels={"importance": "Mean |Attribution Value|", "feature": "Feature"},
        color="importance",
        color_continuous_scale="Teal",
    )
    fig.update_layout(
        paper_bgcolor="#131d31",
        plot_bgcolor="#131d31",
        font=dict(color="#f1f5f9", family="Inter, sans-serif"),
        title=dict(font=dict(color="#f8fafc", size=16)),
        xaxis=dict(
            title=dict(text="Mean |Attribution Value|", font=dict(color="#f8fafc", size=12)),
            tickfont=dict(color="#cbd5e1", size=11),
            gridcolor="#283347",
        ),
        yaxis=dict(
            title=dict(text="Feature", font=dict(color="#f8fafc", size=12)),
            tickfont=dict(color="#cbd5e1", size=11),
            gridcolor="#283347",
        ),
        coloraxis_showscale=False,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def plot_shap_beeswarm(shap_values, max_display: int = 15) -> go.Figure:
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:max_display]

    feature_names = [shap_values.feature_names[i] for i in top_idx]
    shap_subset = shap_values.values[:, top_idx]
    feature_subset = shap_values.data[:, top_idx]

    fig = go.Figure()
    for i, fname in enumerate(reversed(feature_names)):
        col_idx = max_display - 1 - i
        sv = shap_subset[:, col_idx]
        fv = feature_subset[:, col_idx]
        fv_norm = (fv - fv.min()) / (fv.max() - fv.min() + 1e-9)

        fig.add_trace(
            go.Scatter(
                x=sv,
                y=[fname] * len(sv),
                mode="markers",
                marker=dict(
                    size=5,
                    color=fv_norm,
                    colorscale="RdBu_r",
                    opacity=0.7,
                ),
                showlegend=False,
            )
        )

    fig.update_layout(
        paper_bgcolor="#131d31",
        plot_bgcolor="#131d31",
        font=dict(color="#f1f5f9", family="Inter, sans-serif"),
        title=dict(text=f"Feature Impact Distribution (Top {max_display} Features)", font=dict(color="#f8fafc", size=16)),
        xaxis=dict(
            title=dict(text="Attribution Impact on Forecast", font=dict(color="#f8fafc", size=12)),
            tickfont=dict(color="#cbd5e1", size=11),
            gridcolor="#283347",
        ),
        yaxis=dict(
            title=dict(text="Feature", font=dict(color="#f8fafc", size=12)),
            tickfont=dict(color="#cbd5e1", size=11),
            gridcolor="#283347",
        ),
        margin=dict(l=20, r=20, t=60, b=20),
        height=500,
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#94a3b8", line_width=1)
    return fig


def plot_shap_waterfall(shap_values, sample_idx: int = 0) -> go.Figure:
    sample_idx = max(0, min(sample_idx, len(shap_values.values) - 1))
    sv = shap_values.values[sample_idx]
    base = float(shap_values.base_values[sample_idx]) if hasattr(shap_values.base_values, '__len__') else float(shap_values.base_values)
    feature_names = shap_values.feature_names

    order = np.argsort(np.abs(sv))[::-1][:12]
    names = [feature_names[i] for i in order]
    values = sv[order]

    cumulative = np.cumsum(values)
    starts = np.concatenate([[base], base + cumulative[:-1]])

    colors = ["#ef4444" if v > 0 else "#38bdf8" for v in values]

    fig = go.Figure(
        go.Bar(
            x=names,
            y=values,
            base=starts,
            marker_color=colors,
            text=[f"{v:+.3f}" for v in values],
            textposition="outside",
            textfont=dict(color="#f8fafc", size=11),
        )
    )

    fig.add_hline(
        y=base,
        line_dash="dot",
        line_color="#94a3b8",
        annotation_text=f"Base: {base:.3f}",
        annotation_position="top left",
        annotation_font=dict(color="#f8fafc", size=12),
    )

    fig.update_layout(
        paper_bgcolor="#131d31",
        plot_bgcolor="#131d31",
        font=dict(color="#f1f5f9", family="Inter, sans-serif"),
        title=dict(text=f"Feature Contribution Waterfall — Sample #{sample_idx}", font=dict(color="#f8fafc", size=16)),
        xaxis=dict(
            title=dict(text="Feature", font=dict(color="#f8fafc", size=12)),
            tickfont=dict(color="#cbd5e1", size=11),
            gridcolor="#283347",
        ),
        yaxis=dict(
            title=dict(text="Attribution (kWh scaled)", font=dict(color="#f8fafc", size=12)),
            tickfont=dict(color="#cbd5e1", size=11),
            gridcolor="#283347",
        ),
        margin=dict(l=20, r=20, t=60, b=40),
        showlegend=False,
    )
    return fig


def plot_actual_vs_predicted(pred_df: pd.DataFrame) -> go.Figure:
    act_col = "actual_next_hour" if "actual_next_hour" in pred_df.columns else pred_df.columns[0]
    pred_col = "predicted_next_hour" if "predicted_next_hour" in pred_df.columns else pred_df.columns[1]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=pred_df.index,
            y=pred_df[act_col],
            name="Actual",
            line=dict(color="#38bdf8", width=1.8),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pred_df.index,
            y=pred_df[pred_col],
            name="Predicted",
            line=dict(color="#f472b6", width=1.8, dash="dot"),
        )
    )
    fig.update_layout(
        paper_bgcolor="#131d31",
        plot_bgcolor="#131d31",
        font=dict(color="#f1f5f9", family="Inter, sans-serif"),
        title=dict(text="Actual vs Predicted Energy Consumption — Next Hour", font=dict(color="#f8fafc", size=16)),
        xaxis=dict(
            title=dict(text="Time / Window Index", font=dict(color="#f8fafc", size=12)),
            tickfont=dict(color="#cbd5e1", size=11),
            gridcolor="#283347",
        ),
        yaxis=dict(
            title=dict(text="Energy (kWh)", font=dict(color="#f8fafc", size=12)),
            tickfont=dict(color="#cbd5e1", size=11),
            gridcolor="#283347",
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#f1f5f9", size=12),
        ),
        margin=dict(l=20, r=20, t=80, b=20),
    )
    return fig


def plot_residuals(pred_df: pd.DataFrame) -> go.Figure:
    act_col = "actual_next_hour" if "actual_next_hour" in pred_df.columns else pred_df.columns[0]
    pred_col = "predicted_next_hour" if "predicted_next_hour" in pred_df.columns else pred_df.columns[1]
    residuals = pred_df[act_col] - pred_df[pred_col]

    fig = px.histogram(
        residuals,
        nbins=60,
        title="Residuals Distribution (Actual − Predicted)",
        labels={"value": "Residual (kWh)", "count": "Frequency"},
    )
    fig.update_traces(
        marker_color="#10b981", marker_line_color="#1e293b", marker_line_width=0.5
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#ef4444", line_width=1.5)
    fig.update_layout(
        paper_bgcolor="#131d31",
        plot_bgcolor="#131d31",
        font=dict(color="#f1f5f9", family="Inter, sans-serif"),
        title=dict(text="Residuals Distribution (Actual − Predicted)", font=dict(color="#f8fafc", size=16)),
        xaxis=dict(
            title=dict(text="Residual (kWh)", font=dict(color="#f8fafc", size=12)),
            tickfont=dict(color="#cbd5e1", size=11),
            gridcolor="#283347",
        ),
        yaxis=dict(
            title=dict(text="Frequency", font=dict(color="#f8fafc", size=12)),
            tickfont=dict(color="#cbd5e1", size=11),
            gridcolor="#283347",
        ),
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=False,
    )
    return fig


def calculate_carbon_footprint(predicted_kwh: float, emission_factor: float = None) -> dict:
    ef = emission_factor if emission_factor is not None else EMISSION_FACTOR_KG_PER_KWH
    kg_co2 = predicted_kwh * ef
    trees_offset = kg_co2 / 21.77
    km_driven = kg_co2 / 0.21

    return {
        "predicted_kwh": round(float(predicted_kwh), 4),
        "emission_factor": ef,
        "kg_co2": round(float(kg_co2), 4),
        "trees_to_offset": round(float(trees_offset), 2),
        "equivalent_km_driven": round(float(km_driven), 2),
    }


def plot_carbon_gauge(predicted_kwh: float, emission_factor: float = None) -> go.Figure:
    ef = emission_factor if emission_factor is not None else EMISSION_FACTOR_KG_PER_KWH
    carbon = calculate_carbon_footprint(predicted_kwh, emission_factor=ef)
    kg = carbon["kg_co2"]

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=kg,
            title={"text": "Carbon Footprint (kg CO₂)", "font": {"size": 18}},
            delta={"reference": ef * 5, "valueformat": ".2f"},
            gauge={
                "axis": {"range": [0, ef * 25]},
                "bar": {"color": "#d62728"},
                "steps": [
                    {"range": [0, EMISSION_FACTOR_KG_PER_KWH * 5], "color": "#2ca02c"},
                    {
                        "range": [
                            EMISSION_FACTOR_KG_PER_KWH * 5,
                            EMISSION_FACTOR_KG_PER_KWH * 15,
                        ],
                        "color": "#ffbf00",
                    },
                    {
                        "range": [
                            EMISSION_FACTOR_KG_PER_KWH * 15,
                            EMISSION_FACTOR_KG_PER_KWH * 25,
                        ],
                        "color": "#d62728",
                    },
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.75,
                    "value": kg,
                },
            },
        )
    )
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=300)
    return fig


def run_explainability_pipeline():
    print("Loading processed data and artifacts …")
    df = load_processed()
    fe = add_calendar_features(df)
    scaled, feature_cols, feature_scaler, target_scaler, train_end, val_end = fit_and_scale(fe)
    X_train, y_train, X_val, y_val, X_test, y_test = build_windows(scaled, feature_cols, train_end, val_end)

    model, _, _, config, metrics = load_model_artifacts()
    all_feature_names = config["feature_cols"] + [config["target_col"]]

    metrics_out, y_pred, y_true = evaluate_model(model, X_test, y_test, target_scaler)
    pred_df = get_predictions_df(y_true, y_pred)

    print("Computing feature attributions for PyTorch LSTM …")
    shap_values, _ = compute_shap_values(model, X_test, feature_names=all_feature_names)
    print("Attributions computed.")

    sample_kwh = float(pred_df["predicted_next_hour"].iloc[0])
    carbon = calculate_carbon_footprint(sample_kwh)

    print("\n" + "=" * 50)
    print("  CARBON FOOTPRINT — Sample Prediction")
    print("=" * 50)
    print(f"  Predicted energy   : {carbon['predicted_kwh']} kWh")
    print(f"  Emission factor    : {carbon['emission_factor']} kg CO₂/kWh")
    print(f"  CO₂ emitted        : {carbon['kg_co2']} kg")
    print(f"  Trees to offset    : {carbon['trees_to_offset']}")
    print(f"  Equivalent driving : {carbon['equivalent_km_driven']} km")
    print("=" * 50 + "\n")

    return shap_values, pred_df, carbon


if __name__ == "__main__":
    shap_values, pred_df, carbon = run_explainability_pipeline()

    print("Building charts …")
    plot_shap_bar(shap_values).show()
    plot_shap_beeswarm(shap_values).show()
    plot_shap_waterfall(shap_values, sample_idx=0).show()
    plot_actual_vs_predicted(pred_df).show()
    plot_residuals(pred_df).show()
    plot_carbon_gauge(carbon["predicted_kwh"]).show()
    print("All Phase 4 charts shown.")
