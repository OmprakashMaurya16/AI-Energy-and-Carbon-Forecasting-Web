# ⚡ AI Energy & Carbon Forecasting Web

An end-to-end Deep Learning sequence modeling pipeline and interactive web application for multi-horizon household energy consumption forecasting, state-level power grid benchmarking, and carbon emission analytics.

Built with **PyTorch LSTM**, **SHAP Explainable AI**, **Streamlit**, and **Plotly**, covering both micro-level smart meter telemetry (**Mathura & Bareilly 2020**) and macro-level Indian state power dispatch (**POSOCO / Grid-India 2019–2020**).

---

## ⚡ Quick Start (For Group Members)

> [!TIP]
> **No training required to test!** Pre-trained model weights (`models/lstm_energy.pt`) and processed benchmark datasets (`data/processed/`) are already included in this branch. You can clone and run the interactive dashboard in under 3 minutes.

### 1. Clone the repository & switch to branch
```bash
git clone https://github.com/OmprakashMaurya16/AI-Energy-and-Carbon-Forecasting-Web.git
cd AI-Energy-and-Carbon-Forecasting-Web
git checkout Sujal
```

### 2. Set Up Virtual Environment

> [!IMPORTANT]
> **Recommended Python Version**: **Python 3.10 to 3.12**.
> PyTorch and SHAP have stable pre-built wheels for Python 3.12 on Windows.

* **Windows (PowerShell):**
  ```powershell
  # If you have multiple Python versions installed, use Python 3.12:
  py -3.12 -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
  *(If PowerShell gives a script execution policy error, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

* **macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Web Dashboard
```bash
streamlit run app.py
```
Your default web browser will automatically open: **`http://localhost:8501`**.

---

## 🌟 Key Features & Dashboard Walkthrough

The web dashboard is organized into 6 tabs with interactive widgets and plain-English guidance:

### 📍 Dynamic Region & Year Switcher (Sidebar)
* Select between **Mathura (2020)**, **Bareilly (2020)**, or any of **33 Indian States & Union Territories (2019–2020)**.
* Shows dataset origin, collection year, and regional **Central Electricity Authority (CEA)** $\text{CO}_2$ emission factor ($0.45 - 0.96\text{ kg CO}_2/\text{kWh}$).

### 📖 Plain-English Glossary (Sidebar)
* Clickable expander in the sidebar defining domain terms in everyday words:
  * **kWh & MU**: Unit of electricity (1 MU = 1,000,000 kWh).
  * **LSTM**: Deep learning recurrent neural network specialized in sequence memory.
  * **SHAP / XAI**: Feature attribution explaining *why* the model made a forecast.
  * **CEA Baseline**: Government emissions factor converting electricity used into $\text{kg CO}_2$.
  * **MAE / RMSE / MAPE / R²**: Error metrics explaining accuracy in simple terms.

### 📑 The 6 Application Tabs
1. **Executive Overview**: High-level KPIs, 24-hour forecasted energy, dynamic carbon emission gauge, equivalent EV kilometers, and tree offset equivalencies.
2. **24h Forecast Curve**: 168-hour historical context + 24-hour ahead multi-step prediction curves with confidence intervals and zoom controls.
3. **Explainability (XAI)**: SHAP Beeswarm, feature importance bar charts, and single-step waterfall breakdown illustrating the influence of temperature, humidity, voltage, and calendar cycles.
4. **Data Explorer & Heatmaps**: Interactive 2D hour-by-month load heatmaps, seasonal profiles, and data table filtering.
5. **Model Performance**: Actual vs. Predicted scatter plot with $R^2$ fit line, residual error distributions, and error metrics table.
6. **State Comparison (Faculty Validation Benchmark)**:
   * **Multi-State Accuracy Leaderboard**: Compare model generalization across 10 key states (Maharashtra, Gujarat, Uttar Pradesh, Tamil Nadu, Karnataka, Delhi, Rajasthan, West Bengal, Punjab, Kerala).
   * **Power Demand vs. Temperature Correlation**: Scatter plot with OLS trendline showing temperature sensitivity across regions.
   * **CEA Carbon Emission Intensity Comparison**: Color-coded bar chart comparing coal-heavy vs. renewable-rich state emission intensities.

---

## 📂 Project Architecture

```text
AI-Energy-and-Carbon-Forecasting-Web/
├── data/
│   ├── raw/                                ← Raw datasets (CEEW smart meter data, gitignored >100MB)
│   │   ├── CEEW - Smart meter data Bareilly 2020.csv
│   │   └── weather.csv
│   └── processed/                          ← Lightweight processed datasets (committed)
│       ├── merged_hourly.csv               ← Cleaned hourly smart meter + weather dataset
│       ├── posoco_states.csv               ← POSOCO 33-state daily energy consumption (2019–2020)
│       └── state_weather.csv               ← Multi-state historical daily weather telemetry
│
├── src/                                    ← Core modules
│   ├── __init__.py
│   ├── data_engineering.py                 ← 3-min to 1-hr gapless aggregation & feature engineering
│   ├── eda_visualization.py                ← Heatmap & seasonality visualizer
│   ├── modeling.py                         ← PyTorch LSTM sequence model training
│   ├── explainability.py                   ← Deep feature attribution (Gradient × Input & SHAP)
│   ├── multi_state_data.py                 ← POSOCO data cleaner & Open-Meteo state weather fetcher
│   └── multi_state_modeling.py             ← Multi-state sequence forecasting benchmark
│
├── scripts/
│   └── download_weather.py                 ← Automated Open-Meteo hourly weather downloader for UP
│
├── models/                                 ← Saved PyTorch weights & benchmark JSONs
│   ├── lstm_energy.pt                      ← Trained PyTorch LSTM weights (~840 KB)
│   ├── feature_scaler.pkl                  ← Input feature scaler
│   ├── target_scaler.pkl                   ← Target energy scaler
│   ├── model_config.json                   ← Hyperparameters (seq_len=168, horizon=24, hidden=128)
│   ├── metrics.json                        ← Evaluation metrics (MAE, RMSE, MAPE, R²)
│   ├── multi_state_metrics.json            ← 10-state benchmark metrics leaderboard
│   └── multi_state_predictions.json        ← Pre-computed state test predictions
│
├── app.py                                  ← Streamlit Web Application (6 Tabs + Glossary)
├── requirements.txt                        ← Python package requirements
├── .gitignore                              ← Ignores >300MB raw files, tracks lightweight models
└── README.md
```

---

## 🔬 Model Architecture & Mathematical Formulation

```text
[Input Sequence: 168 Hours × 15 Features]
       │
       ▼
[Layer 1: LSTM (hidden_size=128, dropout=0.2)]
       │
       ▼
[Layer 2: LSTM (hidden_size=128)]
       │
       ▼
[Linear Dense Head: 128 -> 24]
       │
       ▼
[Forecast Output: 24 Consecutive Hours]
```

* **Sequence Framing**: Lookback window of $T=168\text{ hours}$ (1 full week) to capture weekly and daily seasonality.
* **Forecast Horizon**: Direct multi-step output of $H=24\text{ hours}$ ahead.
* **Engineered Feature Set (15 Dimensions)**:
  * **Electrical**: `voltage_v`, `current_a`, `frequency_hz`, and autoregressive `energy_kwh` lag.
  * **Weather Telemetry**: `temperature_c`, `humidity_pct`, `precipitation_mm`, `cloud_cover_pct`, `solar_radiation_wm2`, `wind_speed_ms`.
  * **Temporal & Cyclical**: $\sin(2\pi h / 24)$, $\cos(2\pi h / 24)$, $\sin(2\pi m / 12)$, $\cos(2\pi m / 12)$, `is_weekend`.

---

## 🔄 Retraining Pipelines from Scratch (Optional)

If you wish to re-train the models or re-generate the datasets from scratch:

```bash
# 1. Download Open-Meteo hourly weather data for 2020
python scripts/download_weather.py

# 2. Process raw smart meter data into gapless hourly dataset
# (Ensure CEEW CSV is placed in data/raw/)
python src/data_engineering.py

# 3. Train the PyTorch LSTM model (saves weights to models/lstm_energy.pt)
python src/modeling.py

# 4. Fetch multi-state weather and compute 10-state validation benchmarks
python src/multi_state_modeling.py
```

---

## ❓ Frequently Asked Questions (FAQ) & Troubleshooting

### 1. `ModuleNotFoundError: No module named 'torch'`
Make sure your virtual environment is activated before running:
```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Python 3.14 wheel errors
If you are on Python 3.14, PyTorch wheels may not be available yet. Install Python 3.12 and create your virtual environment with:
```powershell
py -3.12 -m venv venv
```

### 3. Port already in use (`Port 8501 is already in use`)
Run Streamlit on a different port:
```bash
streamlit run app.py --server.port 8502
```

---

## 📜 Acknowledgements & Data Sources

* **CEEW Smart Meter Data (Mathura & Bareilly 2020)**: Council on Energy, Environment and Water.
* **POSOCO / GRID-INDIA National Power Dispatch**: Power System Operation Corporation daily state energy reports (2019–2020).
* **Open-Meteo Historical Weather API**: High-resolution hourly ERA5 reanalysis and historical weather telemetry.
* **Central Electricity Authority (CEA)**: Ministry of Power, Government of India — $\text{CO}_2$ Baseline Database for the Indian Power Sector.
