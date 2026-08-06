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
    build_features,
    evaluate_model,
    get_predictions_df,
    load_model_artifacts,
    split_data,
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
    background-color: #0f1117;
    color: #e0e0e0;
}

section[data-testid="stSidebar"] {
    background-color: #161b27;
    border-right: 1px solid #1f2937;
}

.kpi-box {
    background: #161b27;
    border: 1px solid #1f2937;
    border-radius: 10px;
    padding: 20px 24px;
    text-align: center;
}

.kpi-label {
    font-size: 0.75rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
}

.kpi-value {
    font-size: 1.9rem;
    font-weight: 700;
    color: #f9fafb;
    line-height: 1.1;
}

.kpi-sub {
    font-size: 0.72rem;
    color: #4b5563;
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

.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #161b27;
    border-radius: 10px;
    padding: 4px;
    border: 1px solid #1f2937;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 7px;
    padding: 8px 20px;
    font-size: 0.85rem;
    font-weight: 500;
    color: #6b7280;
    background: transparent;
}

.stTabs [aria-selected="true"] {
    background: #1f2937 !important;
    color: #f9fafb !important;
}

div[data-testid="stMetric"] {
    background: #161b27;
    border: 1px solid #1f2937;
    border-radius: 10px;
    padding: 16px 20px;
}

div[data-testid="stMetricLabel"] { color: #6b7280; font-size: 0.78rem; }
div[data-testid="stMetricValue"] { color: #f9fafb; font-size: 1.6rem; font-weight: 700; }

.stSelectbox > div > div {
    background: #161b27;
    border: 1px solid #1f2937;
    color: #e0e0e0;
}

.stSlider > div > div { background: #1f2937; }

hr { border-color: #1f2937; }

.page-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: #f9fafb;
    margin-bottom: 2px;
}

.page-sub {
    font-size: 0.85rem;
    color: #6b7280;
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


@st.cache_data(show_spinner="Building features …")
def get_feature_matrix(_df):
    fe = build_features(_df)
    return split_data(fe)


@st.cache_data(show_spinner="Running predictions …")
def get_predictions(_X_test, _y_test, _model, _scaler):
    _, y_pred = evaluate_model(_model, _scaler, _X_test, _y_test)
    return get_predictions_df(_X_test, _y_test, y_pred), y_pred


@st.cache_data(show_spinner="Computing SHAP …")
def get_shap(_model, _scaler, _X_test):
    return compute_shap_values(_model, _scaler, _X_test)


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


def tab_forecast(df, model, scaler, X_test, y_test):
    st.markdown('<div class="page-title">Energy Forecast</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-sub">Mathura 2020 · 38 Smart Meters · Hourly Prediction</div>',
        unsafe_allow_html=True,
    )

    pred_df, _ = get_predictions(X_test, y_test, model, scaler)

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
            y=sample["actual"],
            name="Actual",
            line=dict(color="#60a5fa", width=2),
            fill="tozeroy",
            fillcolor="rgba(96,165,250,0.06)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sample.index,
            y=sample["predicted"],
            name="Predicted",
            line=dict(color="#f472b6", width=2, dash="dot"),
        )
    )
    fig.update_layout(
        title="Actual vs Predicted — Last 7 Days of Test Set",
        title_font=dict(size=15, color="#e0e0e0"),
        paper_bgcolor="#161b27",
        plot_bgcolor="#161b27",
        font_color="#9ca3af",
        xaxis=dict(gridcolor="#1f2937", showgrid=True),
        yaxis=dict(gridcolor="#1f2937", showgrid=True, title="kWh"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=50, b=10),
        hovermode="x unified",
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    fig2 = px.scatter(
        pred_df,
        x="actual",
        y="predicted",
        title="Scatter: Actual vs Predicted (Full Test Set)",
        labels={"actual": "Actual kWh", "predicted": "Predicted kWh"},
        opacity=0.4,
        color_discrete_sequence=["#a78bfa"],
    )
    max_val = float(max(pred_df["actual"].max(), pred_df["predicted"].max())) + 1
    fig2.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=max_val,
        y1=max_val,
        line=dict(color="#4b5563", width=1, dash="dash"),
    )
    fig2.update_layout(
        paper_bgcolor="#161b27",
        plot_bgcolor="#161b27",
        font_color="#9ca3af",
        xaxis=dict(gridcolor="#1f2937"),
        yaxis=dict(gridcolor="#1f2937"),
        margin=dict(l=10, r=10, t=50, b=10),
        height=350,
        title_font=dict(size=15, color="#e0e0e0"),
    )
    st.plotly_chart(fig2, use_container_width=True)


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
    fig.update_layout(
        paper_bgcolor="#161b27",
        plot_bgcolor="#161b27",
        font_color="#9ca3af",
        title_font=dict(color="#e0e0e0", size=15),
        xaxis=dict(gridcolor="#1f2937"),
        yaxis=dict(gridcolor="#1f2937"),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


def tab_model(df, model, scaler, X_test, y_test):
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
        <div class="kpi-label">Overall Model Accuracy</div>
        <div style="font-size:1.1rem; margin-top:6px; color:#e0e0e0;">
            R² Score: {badge} &nbsp;·&nbsp;
            The model explains <strong style="color:#f9fafb">{round(r2 * 100, 1)}%</strong> of variation in energy consumption.
            On average it is off by <strong style="color:#f9fafb">{mae} kWh</strong> per hour.
            MAPE of <strong style="color:#fbbf24">{mape}%</strong> — predictions deviate {mape}% from actual on average.
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² Score", r2, help="1.0 = perfect. 0.81 = good for energy data.")
    c2.metric("MAE", f"{mae} kWh", help="Average absolute error per prediction.")
    c3.metric("RMSE", f"{rmse} kWh", help="Penalises large errors more than MAE.")
    c4.metric(
        "MAPE", f"{mape} %", help="% error relative to actual value. Lower is better."
    )

    st.markdown("<br>", unsafe_allow_html=True)

    pred_df, _ = get_predictions(X_test, y_test, model, scaler)

    tab_a, tab_b = st.tabs(["Actual vs Predicted", "Error Distribution"])
    with tab_a:
        fig = plot_actual_vs_predicted(pred_df)
        fig.update_layout(
            paper_bgcolor="#161b27",
            plot_bgcolor="#161b27",
            font_color="#9ca3af",
            title_font=dict(color="#e0e0e0", size=15),
            xaxis=dict(gridcolor="#1f2937"),
            yaxis=dict(gridcolor="#1f2937"),
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
    with tab_b:
        fig = plot_residuals(pred_df)
        fig.update_layout(
            paper_bgcolor="#161b27",
            plot_bgcolor="#161b27",
            font_color="#9ca3af",
            title_font=dict(color="#e0e0e0", size=15),
            xaxis=dict(gridcolor="#1f2937"),
            yaxis=dict(gridcolor="#1f2937"),
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)


def tab_explainability(df, model, scaler, X_test, y_test):
    st.markdown(
        '<div class="page-title">Why Did the Model Predict This?</div>',
        unsafe_allow_html=True,
    )

    shap_values, _ = get_shap(model, scaler, X_test)

    view = st.radio(
        "",
        ["Feature Importance", "Beeswarm", "Single Prediction Breakdown"],
        horizontal=True,
        label_visibility="collapsed",
    )

    def apply_dark(fig):
        fig.update_layout(
            paper_bgcolor="#161b27",
            plot_bgcolor="#161b27",
            font_color="#9ca3af",
            title_font=dict(color="#e0e0e0", size=15),
            xaxis=dict(gridcolor="#1f2937"),
            yaxis=dict(gridcolor="#1f2937"),
            margin=dict(l=10, r=10, t=50, b=10),
        )
        return fig

    if view == "Feature Importance":
        top_n = st.slider("Show top N features", 5, 29, 15)
        st.plotly_chart(
            apply_dark(plot_shap_bar(shap_values, top_n=top_n)),
            use_container_width=True,
        )

    elif view == "Beeswarm":
        st.plotly_chart(
            apply_dark(plot_shap_beeswarm(shap_values)), use_container_width=True
        )

    elif view == "Single Prediction Breakdown":
        pred_df, y_pred = get_predictions(X_test, y_test, model, scaler)
        sample_idx = st.slider("Pick a prediction", 0, len(X_test) - 1, 0)

        actual_val = float(y_test.iloc[sample_idx])
        pred_val = float(y_pred[sample_idx])
        error = round(abs(actual_val - pred_val), 4)
        ts = X_test.index[sample_idx]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Timestamp", ts.strftime("%d %b %Y %H:%M"))
        c2.metric("Actual", f"{actual_val:.3f} kWh")
        c3.metric("Predicted", f"{pred_val:.3f} kWh")
        c4.metric("Error", f"{error} kWh")

        st.plotly_chart(
            apply_dark(plot_shap_waterfall(shap_values, sample_idx=sample_idx)),
            use_container_width=True,
        )


def tab_carbon(df, model, scaler, X_test, y_test):
    st.markdown(
        '<div class="page-title">Carbon Footprint</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="page-sub">Based on India UP grid · 0.82 kg CO₂ per kWh · CEA 2022</div>',
        unsafe_allow_html=True,
    )

    pred_df, _ = get_predictions(X_test, y_test, model, scaler)

    kwh_val = st.slider(
        "Adjust kWh to calculate CO₂",
        min_value=0.0,
        max_value=float(df["energy_kwh"].max()),
        value=float(pred_df["predicted"].mean()),
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
    fig_gauge.update_layout(
        paper_bgcolor="#161b27",
        font_color="#9ca3af",
        margin=dict(l=10, r=10, t=20, b=10),
        height=280,
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    carbon_df = pred_df.copy()
    carbon_df["co2_kg"] = carbon_df["predicted"] * EMISSION_FACTOR_KG_PER_KWH
    carbon_df["co2_cumulative_kg"] = carbon_df["co2_kg"].cumsum()

    total_co2 = carbon_df["co2_kg"].sum()

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi(
            "Total CO₂ (Test Period)",
            f"{total_co2:,.0f} kg",
            "Oct–Dec 2020 predictions",
        )
    with c2:
        kpi("Equiv. Driving", f"{total_co2 / 0.21:,.0f} km", "At 0.21 kg CO₂/km")
    with c3:
        kpi("Trees for 1 Year", f"{total_co2 / 21.77:,.0f}", "To fully offset")

    st.markdown("<br>", unsafe_allow_html=True)

    fig_area = px.area(
        carbon_df,
        y="co2_cumulative_kg",
        title="Cumulative CO₂ Emissions — Test Period (Oct–Dec 2020)",
        labels={"co2_cumulative_kg": "Cumulative CO₂ (kg)"},
        color_discrete_sequence=["#34d399"],
    )
    fig_area.update_layout(
        paper_bgcolor="#161b27",
        plot_bgcolor="#161b27",
        font_color="#9ca3af",
        title_font=dict(color="#e0e0e0", size=15),
        xaxis=dict(gridcolor="#1f2937"),
        yaxis=dict(gridcolor="#1f2937"),
        margin=dict(l=10, r=10, t=50, b=10),
        height=340,
    )
    st.plotly_chart(fig_area, use_container_width=True)


def main():
    model_ready = os.path.exists(MODEL_PATH)

    with st.sidebar:
        st.markdown("### ⚡ Energy Forecaster")
        st.markdown(
            '<div style="color:#6b7280; font-size:0.8rem; margin-bottom:16px;">Mathura 2020 · XGBoost + SHAP</div>',
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
            '<div class="warn-banner">⚠️ Model not found. Run <code>python -m src.modeling</code> then restart.</div>',
            unsafe_allow_html=True,
        )

    df = get_data()

    if not model_ready:
        tab_eda(df)
        return

    model, scaler, metrics = get_model_artifacts()
    X_train, X_test, y_train, y_test, feature_cols = get_feature_matrix(df)

    t1, t2, t3, t4, t5 = st.tabs(
        ["Forecast", "Data Analysis", "Model", "Explainability", "Carbon"]
    )

    with t1:
        tab_forecast(df, model, scaler, X_test, y_test)
    with t2:
        tab_eda(df)
    with t3:
        tab_model(df, model, scaler, X_test, y_test)
    with t4:
        tab_explainability(df, model, scaler, X_test, y_test)
    with t5:
        tab_carbon(df, model, scaler, X_test, y_test)


if __name__ == "__main__":
    main()
