import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

PROCESSED_PATH = os.path.join(_ROOT, "data", "processed", "merged_hourly.csv")


def load_processed() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_PATH, parse_dates=["datetime"], index_col="datetime")
    return df


def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    corr = df.select_dtypes(include="number").corr().round(2)

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Feature Correlation Heatmap",
        aspect="auto",
    )
    fig.update_layout(
        title_font_size=18,
        coloraxis_colorbar=dict(title="r"),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def plot_energy_vs_temperature(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["energy_kwh"],
            name="Energy (kWh)",
            line=dict(color="#1f77b4", width=1),
            opacity=0.85,
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["temperature_c"],
            name="Temperature (°C)",
            line=dict(color="#d62728", width=1, dash="dot"),
            opacity=0.75,
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title="Hourly Energy Consumption vs Temperature — 2020",
        title_font_size=18,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=80, b=20),
    )
    fig.update_yaxes(title_text="Energy (kWh)", secondary_y=False)
    fig.update_yaxes(title_text="Temperature (°C)", secondary_y=True)
    fig.update_xaxes(title_text="Datetime")
    return fig


def plot_daily_seasonality(df: pd.DataFrame) -> go.Figure:
    hourly_avg = df.groupby(df.index.hour)["energy_kwh"].mean().reset_index()
    hourly_avg.columns = ["hour", "avg_energy_kwh"]

    fig = px.bar(
        hourly_avg,
        x="hour",
        y="avg_energy_kwh",
        title="Average Hourly Energy Consumption (Daily Seasonality)",
        labels={"hour": "Hour of Day", "avg_energy_kwh": "Avg Energy (kWh)"},
        color="avg_energy_kwh",
        color_continuous_scale="Blues",
    )
    fig.update_layout(
        title_font_size=18,
        coloraxis_showscale=False,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis=dict(tickmode="linear", tick0=0, dtick=1),
    )
    return fig


def plot_monthly_energy(df: pd.DataFrame) -> go.Figure:
    monthly = df.groupby(df.index.month)["energy_kwh"].sum().reset_index()
    monthly.columns = ["month", "total_energy_kwh"]
    month_names = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    monthly["month_name"] = monthly["month"].apply(lambda x: month_names[x - 1])

    fig = px.line(
        monthly,
        x="month_name",
        y="total_energy_kwh",
        markers=True,
        title="Total Monthly Energy Consumption — 2020",
        labels={"month_name": "Month", "total_energy_kwh": "Total Energy (kWh)"},
    )
    fig.update_traces(line=dict(color="#2ca02c", width=2), marker=dict(size=8))
    fig.update_layout(
        title_font_size=18,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def plot_energy_distribution(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df,
        x="energy_kwh",
        nbins=80,
        title="Distribution of Hourly Energy Consumption",
        labels={"energy_kwh": "Energy (kWh)", "count": "Frequency"},
    )
    fig.update_traces(
        marker_color="#1f77b4", marker_line_color="white", marker_line_width=0.5
    )
    fig.update_layout(
        title_font_size=18,
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=False,
    )
    return fig


def plot_weekly_heatmap(df: pd.DataFrame) -> go.Figure:
    pivot = df.copy()
    pivot["hour"] = pivot.index.hour
    pivot["dayofweek"] = pivot.index.dayofweek
    heat = pivot.groupby(["dayofweek", "hour"])["energy_kwh"].mean().unstack()

    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    fig = px.imshow(
        heat,
        labels=dict(x="Hour of Day", y="Day of Week", color="Avg kWh"),
        x=list(range(24)),
        y=day_labels,
        color_continuous_scale="YlOrRd",
        title="Weekly Energy Heatmap (Avg kWh by Day × Hour)",
        aspect="auto",
    )
    fig.update_layout(
        title_font_size=18,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def run_all_charts(df: pd.DataFrame) -> None:
    print("Generating Chart 1/6 — Correlation Heatmap")
    plot_correlation_heatmap(df).show()

    print("Generating Chart 2/6 — Energy vs Temperature Time-Series")
    plot_energy_vs_temperature(df).show()

    print("Generating Chart 3/6 — Daily Seasonality")
    plot_daily_seasonality(df).show()

    print("Generating Chart 4/6 — Monthly Energy")
    plot_monthly_energy(df).show()

    print("Generating Chart 5/6 — Energy Distribution")
    plot_energy_distribution(df).show()

    print("Generating Chart 6/6 — Weekly Heatmap")
    plot_weekly_heatmap(df).show()

    print("\nAll 6 charts generated successfully.")


if __name__ == "__main__":
    df = load_processed()
    print(f"Loaded processed data: {df.shape[0]} rows × {df.shape[1]} cols")
    run_all_charts(df)
