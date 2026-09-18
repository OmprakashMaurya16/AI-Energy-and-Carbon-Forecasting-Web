import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_engineering import EMISSION_FACTOR_KG_PER_KWH, load_and_prepare_data
from src.eda_visualization import (
    plot_correlation_heatmap,
    plot_daily_seasonality,
    plot_energy_distribution,
    plot_energy_vs_temperature,
    plot_monthly_energy,
    plot_weekly_heatmap,
)
from src.explainability import (
    calculate_carbon_footprint,
    compute_shap_values,
    plot_actual_vs_predicted,
    plot_carbon_gauge,
    plot_residuals,
    plot_shap_bar,
    plot_shap_beeswarm,
    plot_shap_waterfall,
)
from src.modeling import (
    MODEL_PATH,
    add_calendar_features,
    build_windows,
    evaluate_model,
    fit_and_scale,
    get_predictions_df,
    load_model_artifacts,
)

_ROOT = os.path.dirname(os.path.abspath(__file__))
METRICS_PATH = os.path.join(_ROOT, "models", "metrics.json")
PROCESSED_PATH = os.path.join(_ROOT, "data", "processed", "merged_hourly.csv")

st.set_page_config(
    page_title="Energy & Carbon Forecaster",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #0b0f19;
    color: #f1f5f9;
}

section[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #1e293b;
}

section[data-testid="stSidebar"] * {
    color: #e2e8f0;
}

.kpi-box {
    background: #131d31;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 20px 24px;
    text-align: center;
}

.kpi-label {
    font-size: 0.8rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
    font-weight: 600;
}

.kpi-value {
    font-size: 1.9rem;
    font-weight: 700;
    color: #f8fafc;
    line-height: 1.1;
}

.kpi-sub {
    font-size: 0.78rem;
    color: #cbd5e1;
    margin-top: 4px;
}

.accuracy-bad {
    color: #f87171;
    font-weight: 700;
}

.accuracy-ok {
    color: #fbbf24;
    font-weight: 700;
}

.accuracy-good {
    color: #34d399;
    font-weight: 700;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: #111827;
    border-radius: 10px;
    padding: 6px;
    border: 1px solid #1e293b;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 7px;
    padding: 8px 22px;
    font-size: 0.9rem;
    font-weight: 600;
    color: #94a3b8;
    background: transparent;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #f8fafc;
}

.stTabs [aria-selected="true"] {
    background: #1e293b !important;
    color: #38bdf8 !important;
}

/* Metrics */
div[data-testid="stMetric"] {
    background: #131d31;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 16px 20px;
}

div[data-testid="stMetricLabel"],
div[data-testid="stMetricLabel"] * {
    color: #94a3b8 !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
}

div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] * {
    color: #f8fafc !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
}

/* Radio buttons & text */
div[data-testid="stRadio"] label,
div[data-testid="stRadio"] p,
div[data-testid="stRadio"] span,
div[data-testid="stRadio"] div {
    color: #f1f5f9 !important;
    font-size: 0.95rem;
    font-weight: 500;
}

/* Slider labels & ticks */
div[data-testid="stSlider"] label,
div[data-testid="stSlider"] p,
div[data-testid="stSlider"] span,
div[data-testid="stSlider"] div {
    color: #f1f5f9 !important;
    font-size: 0.9rem;
}

div[data-testid="stSlider"] > div > div {
    background: #1e293b;
}

/* Selectbox */
.stSelectbox label,
.stSelectbox p,
.stSelectbox span {
    color: #f1f5f9 !important;
    font-weight: 500;
}

.stSelectbox > div > div {
    background: #131d31;
    border: 1px solid #1e293b;
    color: #f1f5f9;
}

hr {
    border-color: #1e293b;
}

.page-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 2px;
}

.page-sub {
    font-size: 0.9rem;
    color: #94a3b8;
    margin-bottom: 20px;
}

.warn-banner {
    background: #1c1400;
    border-left: 4px solid #f59e0b;
    border-radius: 6px;
    padding: 12px 16px;
    color: #fbbf24;
    font-size: 0.85rem;
    margin-bottom: 16px;
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading data …")
def get_data() -> pd.DataFrame:
    if os.path.exists(PROCESSED_PATH):
        return pd.read_csv(
            PROCESSED_PATH, parse_dates=["datetime"], index_col="datetime"
        )
    return load_and_prepare_data()


@st.cache_resource(show_spinner="Loading model …")
def get_model_artifacts():
    return load_model_artifacts()


@st.cache_data(show_spinner="Building sequences …")
def get_sequences(_df):
    fe = add_calendar_features(_df)
    scaled, feature_cols, feature_scaler, target_scaler, train_end, val_end = fit_and_scale(fe)
    X_train, y_train, X_val, y_val, X_test, y_test = build_windows(
        scaled, feature_cols, train_end, val_end
    )
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols


@st.cache_data(show_spinner="Running predictions …")
def get_predictions(_X_test, _y_test, _model, _target_scaler):
    metrics, y_pred, y_true = evaluate_model(_model, _X_test, _y_test, _target_scaler)
    return get_predictions_df(y_true, y_pred), y_pred, y_true


@st.cache_data(show_spinner="Computing feature attributions …")
def get_shap(_model, _X_test, _feature_names):
    return compute_shap_values(_model, _X_test, feature_names=_feature_names)


def kpi(label, value, sub=""):
    st.markdown(
        f"""
    <div class="kpi-box">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>""",
        unsafe_allow_html=True,
    )


def accuracy_badge(r2: float) -> str:
    if r2 >= 0.85:
        return f'<span class="accuracy-good">Good ({r2})</span>'
    elif r2 >= 0.70:
        return f'<span class="accuracy-ok">Moderate ({r2})</span>'
    else:
        return f'<span class="accuracy-bad">Poor ({r2})</span>'


def apply_dark(fig, height=None):
    fig.update_layout(
        paper_bgcolor="#131d31",
        plot_bgcolor="#131d31",
        font=dict(color="#f1f5f9", family="Inter, sans-serif"),
        title=dict(font=dict(color="#f8fafc", size=16, family="Inter, sans-serif")),
        xaxis=dict(
            gridcolor="#283347",
            tickfont=dict(color="#cbd5e1", size=11),
            title_font=dict(color="#f8fafc", size=12),
        ),
        yaxis=dict(
            gridcolor="#283347",
            tickfont=dict(color="#cbd5e1", size=11),
            title_font=dict(color="#f8fafc", size=12),
        ),
        legend=dict(
            font=dict(color="#f1f5f9", size=11),
        ),
        margin=dict(l=15, r=15, t=50, b=20),
    )
    if height:
        fig.update_layout(height=height)
    return fig


def tab_forecast(df, model, target_scaler, X_test, y_test):
    st.markdown('<div class="page-title">Energy Forecast</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-sub">Mathura 2020 · 38 Smart Meters · 24h Horizon PyTorch LSTM Forecaster</div>',
        unsafe_allow_html=True,
    )

    pred_df, y_pred, y_true = get_predictions(X_test, y_test, model, target_scaler)

    total_kwh = df["energy_kwh"].sum()
    peak_kwh = df["energy_kwh"].max()
    mean_kwh = df["energy_kwh"].mean()
    total_co2 = total_kwh * EMISSION_FACTOR_KG_PER_KWH

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("Total Energy (2020)", f"{total_kwh:,.0f} kWh", "Full year · 38 meters")
    with c2:
        kpi("Avg Hourly Load", f"{mean_kwh:.2f} kWh", "Per hour across all meters")
    with c3:
        kpi("Peak Hourly Load", f"{peak_kwh:.2f} kWh", "Highest single hour in 2020")
    with c4:
        kpi("Total CO₂ Estimated", f"{total_co2:,.0f} kg", "@ 0.82 kg/kWh · UP grid")

    st.markdown("<br>", unsafe_allow_html=True)

    fig = go.Figure()
    sample = pred_df.tail(168)
    fig.add_trace(
        go.Scatter(
            x=sample.index,
            y=sample["actual_next_hour"],
            name="Actual (Next Hour)",
            line=dict(color="#38bdf8", width=2),
            fill="tozeroy",
            fillcolor="rgba(56,189,248,0.08)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sample.index,
            y=sample["predicted_next_hour"],
            name="Predicted (Next Hour)",
            line=dict(color="#f472b6", width=2, dash="dot"),
        )
    )
    fig.update_layout(
        title="Actual vs Predicted — Last 7 Days of Test Windows (Next-Hour Step)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(apply_dark(fig, height=380), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    fig2 = px.scatter(
        pred_df,
        x="actual_next_hour",
        y="predicted_next_hour",
        title="Scatter: Actual vs Predicted (Full Test Set)",
        labels={"actual_next_hour": "Actual kWh", "predicted_next_hour": "Predicted kWh"},
        opacity=0.5,
        color_discrete_sequence=["#a78bfa"],
    )
    max_val = float(max(pred_df["actual_next_hour"].max(), pred_df["predicted_next_hour"].max())) + 1
    fig2.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=max_val,
        y1=max_val,
        line=dict(color="#64748b", width=1.5, dash="dash"),
    )
    st.plotly_chart(apply_dark(fig2, height=350), use_container_width=True)


def tab_eda(df):
    st.markdown('<div class="page-title">Data Analysis</div>', unsafe_allow_html=True)

    chart = st.selectbox(
        "",
        [
            "Energy vs Temperature",
            "Daily Usage Pattern",
            "Monthly Trend",
            "Day × Hour Heatmap",
            "Correlation Heatmap",
            "Consumption Distribution",
        ],
        label_visibility="collapsed",
    )

    charts = {
        "Energy vs Temperature": plot_energy_vs_temperature,
        "Daily Usage Pattern": plot_daily_seasonality,
        "Monthly Trend": plot_monthly_energy,
        "Day × Hour Heatmap": plot_weekly_heatmap,
        "Correlation Heatmap": plot_correlation_heatmap,
        "Consumption Distribution": plot_energy_distribution,
    }

    fig = charts[chart](df)
    st.plotly_chart(apply_dark(fig), use_container_width=True)


def tab_model(df, model, target_scaler, X_test, y_test):
    st.markdown(
        '<div class="page-title">Model Performance</div>', unsafe_allow_html=True
    )

    m = json.load(open(METRICS_PATH))
    r2 = m["R2"]
    mae = m["MAE"]
    rmse = m["RMSE"]
    mape = m["MAPE"]

    badge = accuracy_badge(r2)

    st.markdown(
        f"""
    <div class="kpi-box" style="margin-bottom:20px; text-align:left; padding: 18px 24px;">
        <div class="kpi-label">Overall Model Accuracy (PyTorch LSTM)</div>
        <div style="font-size:1.1rem; margin-top:6px; color:#f1f5f9;">
            R² Score: {badge} &nbsp;·&nbsp;
            Average Test MAE: <strong style="color:#38bdf8">{mae} kWh</strong> per hour.
            RMSE: <strong style="color:#f8fafc">{rmse} kWh</strong>.
            MAPE: <strong style="color:#fbbf24">{mape}%</strong> across 24h prediction horizons.
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² Score", r2, help="1.0 = perfect. Evaluated on 24h multi-horizon test sequences.")
    c2.metric("MAE", f"{mae} kWh", help="Average absolute error across 24h forecast horizons.")
    c3.metric("RMSE", f"{rmse} kWh", help="Root mean squared error.")
    c4.metric(
        "MAPE", f"{mape} %", help="% error relative to actual value. Lower is better."
    )

    st.markdown("<br>", unsafe_allow_html=True)

    pred_df, y_pred, y_true = get_predictions(X_test, y_test, model, target_scaler)

    tab_a, tab_b = st.tabs(["Actual vs Predicted", "Error Distribution"])
    with tab_a:
        fig = plot_actual_vs_predicted(pred_df)
        st.plotly_chart(apply_dark(fig), use_container_width=True)
    with tab_b:
        fig = plot_residuals(pred_df)
        st.plotly_chart(apply_dark(fig), use_container_width=True)


def tab_explainability(df, model, target_scaler, X_test, y_test, all_feature_names):
    st.markdown(
        '<div class="page-title">Why Did the Model Predict This?</div>',
        unsafe_allow_html=True,
    )

    shap_values, _ = get_shap(model, X_test, all_feature_names)

    view = st.radio(
        "",
        ["Feature Importance", "Beeswarm", "Single Prediction Breakdown"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if view == "Feature Importance":
        top_n = st.slider("Show top N features", 3, len(all_feature_names), min(10, len(all_feature_names)))
        st.plotly_chart(
            apply_dark(plot_shap_bar(shap_values, top_n=top_n)),
            use_container_width=True,
        )

    elif view == "Beeswarm":
        st.plotly_chart(
            apply_dark(plot_shap_beeswarm(shap_values)), use_container_width=True
        )

    elif view == "Single Prediction Breakdown":
        pred_df, y_pred, y_true = get_predictions(X_test, y_test, model, target_scaler)
        sample_idx = st.slider("Pick a test window", 0, len(X_test) - 1, 0)

        actual_val = float(pred_df["actual_next_hour"].iloc[sample_idx])
        pred_val = float(pred_df["predicted_next_hour"].iloc[sample_idx])
        error = round(abs(actual_val - pred_val), 4)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Window #", f"#{sample_idx}")
        c2.metric("Actual (Next Hour)", f"{actual_val:.3f} kWh")
        c3.metric("Predicted (Next Hour)", f"{pred_val:.3f} kWh")
        c4.metric("Error", f"{error:.3f} kWh")

        st.plotly_chart(
            apply_dark(plot_shap_waterfall(shap_values, sample_idx=sample_idx)),
            use_container_width=True,
        )


def tab_carbon(df, model, target_scaler, X_test, y_test):
    st.markdown(
        '<div class="page-title">Carbon Footprint</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="page-sub">Based on India UP grid · 0.82 kg CO₂ per kWh · CEA 2022</div>',
        unsafe_allow_html=True,
    )

    pred_df, y_pred, y_true = get_predictions(X_test, y_test, model, target_scaler)

    kwh_val = st.slider(
        "Adjust kWh to calculate CO₂",
        min_value=0.0,
        max_value=float(df["energy_kwh"].max()),
        value=float(pred_df["predicted_next_hour"].mean()),
        step=0.1,
    )

    carbon = calculate_carbon_footprint(kwh_val)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("Energy", f"{carbon['predicted_kwh']} kWh", "Input")
    with c2:
        kpi("CO₂ Emitted", f"{carbon['kg_co2']} kg", "This hour")
    with c3:
        kpi(
            "Trees to Offset",
            f"{carbon['trees_to_offset']}",
            "1 tree = 21.77 kg CO₂/yr",
        )
    with c4:
        kpi(
            "Driving Equiv.",
            f"{carbon['equivalent_km_driven']} km",
            "Avg car = 0.21 kg/km",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    fig_gauge = plot_carbon_gauge(kwh_val)
    st.plotly_chart(apply_dark(fig_gauge, height=280), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    carbon_df = pred_df.copy()
    carbon_df["co2_kg"] = carbon_df["predicted_next_hour"] * EMISSION_FACTOR_KG_PER_KWH
    carbon_df["co2_cumulative_kg"] = carbon_df["co2_kg"].cumsum()

    total_co2 = carbon_df["co2_kg"].sum()

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi(
            "Total CO₂ (Test Period)",
            f"{total_co2:,.0f} kg",
            "Test windows total",
        )
    with c2:
        kpi("Equiv. Driving", f"{total_co2 / 0.21:,.0f} km", "At 0.21 kg CO₂/km")
    with c3:
        kpi("Trees for 1 Year", f"{total_co2 / 21.77:,.0f}", "To fully offset")

    st.markdown("<br>", unsafe_allow_html=True)

    fig_area = px.area(
        carbon_df,
        y="co2_cumulative_kg",
        title="Cumulative CO₂ Emissions — Test Windows",
        labels={"co2_cumulative_kg": "Cumulative CO₂ (kg)"},
        color_discrete_sequence=["#34d399"],
    )
    st.plotly_chart(apply_dark(fig_area, height=340), use_container_width=True)


def main():
    model_ready = os.path.exists(MODEL_PATH)

    with st.sidebar:
        st.markdown("### ⚡ Energy Forecaster")
        st.markdown(
            '<div style="color:#6b7280; font-size:0.8rem; margin-bottom:16px;">Mathura 2020 · PyTorch LSTM + Deep Attribution</div>',
            unsafe_allow_html=True,
        )
        st.divider()

        if model_ready:
            m = json.load(open(METRICS_PATH))
            st.markdown("**Model Accuracy**")
            badge = accuracy_badge(m["R2"])
            st.markdown(f"R² : {badge}", unsafe_allow_html=True)
            st.markdown(f"MAE : `{m['MAE']} kWh`")
            st.markdown(f"MAPE : `{m['MAPE']}%`")
            st.divider()

        st.markdown("**Quick Carbon Calc**")
        kwh_q = st.number_input(
            "kWh", 0.0, 100.0, 6.5, 0.1, format="%.1f", label_visibility="collapsed"
        )
        c = calculate_carbon_footprint(kwh_q)
        st.markdown(f"CO₂ &nbsp;&nbsp;`{c['kg_co2']} kg`")
        st.markdown(f"Trees `{c['trees_to_offset']}`")
        st.markdown(f"Drive `{c['equivalent_km_driven']} km`")

    if not model_ready:
        st.markdown(
            '<div class="warn-banner">⚠️ Model not found. Run <code>python src/modeling.py</code> then restart.</div>',
            unsafe_allow_html=True,
        )

    df = get_data()

    if not model_ready:
        tab_eda(df)
        return

    model, feature_scaler, target_scaler, config, metrics = get_model_artifacts()
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = get_sequences(df)
    all_feature_names = config["feature_cols"] + [config["target_col"]]

    t1, t2, t3, t4, t5 = st.tabs(
        ["Forecast", "Data Analysis", "Model", "Explainability", "Carbon"]
    )

    with t1:
        tab_forecast(df, model, target_scaler, X_test, y_test)
    with t2:
        tab_eda(df)
    with t3:
        tab_model(df, model, target_scaler, X_test, y_test)
    with t4:
        tab_explainability(df, model, target_scaler, X_test, y_test, all_feature_names)
    with t5:
        tab_carbon(df, model, target_scaler, X_test, y_test)


if __name__ == "__main__":
    main()
