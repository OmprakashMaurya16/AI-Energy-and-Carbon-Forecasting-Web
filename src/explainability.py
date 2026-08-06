import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
from xgboost import XGBRegressor

from src.modeling import (
    build_features,
    get_predictions_df,
    load_model_artifacts,
    load_processed,
    split_data,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

EMISSION_FACTOR_KG_PER_KWH = 0.82
METRICS_PATH = os.path.join(_ROOT, "models", "metrics.json")


def compute_shap_values(model: XGBRegressor, scaler, X_test: pd.DataFrame):
    X_scaled = scaler.transform(X_test)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X_test.columns, index=X_test.index)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_scaled_df)
    return shap_values, X_scaled_df


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
        title=f"SHAP Feature Importance (Top {top_n})",
        labels={"importance": "Mean |SHAP Value|", "feature": "Feature"},
        color="importance",
        color_continuous_scale="Teal",
    )
    fig.update_layout(
        title_font_size=18,
        coloraxis_showscale=False,
        margin=dict(l=20, r=20, t=60, b=20),
        yaxis=dict(tickfont=dict(size=11)),
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
                    size=4,
                    color=fv_norm,
                    colorscale="RdBu_r",
                    opacity=0.6,
                ),
                showlegend=False,
            )
        )

    fig.update_layout(
        title=f"SHAP Beeswarm Plot (Top {max_display} Features)",
        title_font_size=18,
        xaxis_title="SHAP Value (impact on model output)",
        yaxis_title="Feature",
        margin=dict(l=20, r=20, t=60, b=20),
        height=500,
    )
    fig.add_vline(x=0, line_dash="dash", line_color="grey", line_width=1)
    return fig


def plot_shap_waterfall(shap_values, sample_idx: int = 0) -> go.Figure:
    sv = shap_values.values[sample_idx]
    base = float(shap_values.base_values[sample_idx])
    feature_names = shap_values.feature_names

    order = np.argsort(np.abs(sv))[::-1][:12]
    names = [feature_names[i] for i in order]
    values = sv[order]

    cumulative = np.cumsum(values)
    starts = np.concatenate([[base], base + cumulative[:-1]])

    colors = ["#d62728" if v > 0 else "#1f77b4" for v in values]

    fig = go.Figure(
        go.Bar(
            x=names,
            y=values,
            base=starts,
            marker_color=colors,
            text=[f"{v:+.3f}" for v in values],
            textposition="outside",
        )
    )

    fig.add_hline(
        y=base,
        line_dash="dot",
        line_color="grey",
        annotation_text=f"Base: {base:.3f}",
        annotation_position="top left",
    )

    fig.update_layout(
        title=f"SHAP Waterfall — Sample #{sample_idx}",
        title_font_size=18,
        xaxis_title="Feature",
        yaxis_title="SHAP Contribution (kWh)",
        margin=dict(l=20, r=20, t=60, b=40),
        showlegend=False,
    )
    return fig


def plot_actual_vs_predicted(pred_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=pred_df.index,
            y=pred_df["actual"],
            name="Actual",
            line=dict(color="#1f77b4", width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pred_df.index,
            y=pred_df["predicted"],
            name="Predicted",
            line=dict(color="#ff7f0e", width=1.5, dash="dot"),
        )
    )
    fig.update_layout(
        title="Actual vs Predicted Energy Consumption — Test Set",
        title_font_size=18,
        xaxis_title="Datetime",
        yaxis_title="Energy (kWh)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=80, b=20),
    )
    return fig


def plot_residuals(pred_df: pd.DataFrame) -> go.Figure:
    residuals = pred_df["actual"] - pred_df["predicted"]
    fig = px.histogram(
        residuals,
        nbins=60,
        title="Residuals Distribution (Actual − Predicted)",
        labels={"value": "Residual (kWh)", "count": "Frequency"},
    )
    fig.update_traces(
        marker_color="#2ca02c", marker_line_color="white", marker_line_width=0.5
    )
    fig.add_vline(x=0, line_dash="dash", line_color="red", line_width=1.5)
    fig.update_layout(
        title_font_size=18,
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=False,
    )
    return fig


def calculate_carbon_footprint(predicted_kwh: float) -> dict:
    kg_co2 = predicted_kwh * EMISSION_FACTOR_KG_PER_KWH
    trees_offset = kg_co2 / 21.77
    km_driven = kg_co2 / 0.21

    return {
        "predicted_kwh": round(predicted_kwh, 4),
        "emission_factor": EMISSION_FACTOR_KG_PER_KWH,
        "kg_co2": round(kg_co2, 4),
        "trees_to_offset": round(trees_offset, 2),
        "equivalent_km_driven": round(km_driven, 2),
    }


def plot_carbon_gauge(predicted_kwh: float) -> go.Figure:
    carbon = calculate_carbon_footprint(predicted_kwh)
    kg = carbon["kg_co2"]

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=kg,
            title={"text": "Carbon Footprint (kg CO₂)", "font": {"size": 18}},
            delta={"reference": EMISSION_FACTOR_KG_PER_KWH * 5, "valueformat": ".2f"},
            gauge={
                "axis": {"range": [0, EMISSION_FACTOR_KG_PER_KWH * 25]},
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
    fe = build_features(df)
    X_train, X_test, y_train, y_test, feature_cols = split_data(fe)
    model, scaler, metrics = load_model_artifacts()

    from src.modeling import evaluate_model

    metrics_out, y_pred = evaluate_model(model, scaler, X_test, y_test)
    pred_df = get_predictions_df(X_test, y_test, y_pred)

    print("Computing SHAP values (may take ~20 seconds) …")
    shap_values, X_scaled_df = compute_shap_values(model, scaler, X_test)
    print("SHAP values computed.")

    sample_kwh = float(pred_df["predicted"].iloc[0])
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
