# ⚡ Explainable Deep Learning Energy & Carbon Forecaster

An end-to-end Deep Learning pipeline and interactive web application for multi-horizon household energy consumption forecasting and carbon footprint estimation using Indian smart meter data (**Mathura 2020**) and local weather features.

---

## 🚀 Key Highlights

* **Deep Learning Sequence Model**: 2-layer **PyTorch LSTM** neural network trained with sliding history windows of **168 hours (1 week)** to predict the next **24 hours** in a single multi-step forward pass.
* **Robust Time-Series Pipeline**: Automated 3-minute to 1-hour aggregation, continuous gapless reindexing, and time-based interpolation across 3.7+ million raw meter readings.
* **Explainable AI (XAI)**: Gradient-based deep feature attribution (`Gradient × Input`) across temporal sequence dimensions, visualizing feature importance with **SHAP Bar, Beeswarm, and Waterfall** breakdown charts.
* **Carbon Emission & Sustainability Analytics**: Real-time conversion of energy demand to $\text{CO}_2$ emissions based on the Indian grid emission baseline (**$0.82\text{ kg CO}_2/\text{kWh}$**), equivalent driving distances, and tree-offset equivalencies.
* **Modern Dark-Themed Web Dashboard**: Built with **Streamlit** and **Plotly**, featuring high-contrast metrics, forecast curves, seasonality analysis, and single-prediction window inspectors.

---

## 🛠️ Tech Stack

| Layer | Technologies |
|---|---|
| **Deep Learning & Modeling** | PyTorch (`torch.nn.LSTM`), Scikit-learn, Joblib |
| **Data Engineering** | Pandas, NumPy |
| **Explainability (XAI)** | SHAP, PyTorch Gradient Attribution |
| **Data Visualization** | Plotly Express & Graph Objects |
| **Web Dashboard** | Streamlit |

---

## 📂 Project Structure

```text
AI-Energy-and-Carbon-Forecasting-Web/
├── data/
│   ├── raw/                                ← Raw input CSVs
│   │   ├── CEEW - Smart meter data Mathura 2020.csv
│   │   └── weather.csv
│   └── processed/                          ← Cleaned, continuous hourly dataset
│       └── merged_hourly.csv
│
├── src/                                    ← Core pipeline source modules
│   ├── __init__.py
│   ├── data_engineering.py                 ← Phase 1: Aggregation, cleaning & interpolation
│   ├── eda_visualization.py                ← Phase 2: Exploratory data analysis & heatmaps
│   ├── modeling.py                         ← Phase 3: PyTorch LSTM training & sequence windowing
│   └── explainability.py                   ← Phase 4: Deep feature attribution & SHAP plots
│
├── models/                                 ← Saved PyTorch model & scalers
│   ├── lstm_energy.pt                      ← Trained LSTM weights
│   ├── feature_scaler.pkl                  ← StandardScaler for input features
│   ├── target_scaler.pkl                   ← StandardScaler for target kWh
│   ├── model_config.json                   ← Model architecture hyperparameters
│   └── metrics.json                        ← Evaluation metrics (MAE, RMSE, MAPE, R²)
│
├── app.py                                  ← Phase 5: Streamlit Web Dashboard
├── requirements.txt                        ← Project dependencies
├── .gitignore
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/OmprakashMaurya16/AI-Energy-and-Carbon-Forecasting-Web.git
cd AI-Energy-and-Carbon-Forecasting-Web
```

### 2. Create and activate virtual environment
* **Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
* **macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🏃 Execution Guide

### Option A: Run the Web Dashboard
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

### Option B: Run Pipeline Modules Individually

```bash
# 1. Prepare and clean data (3-min -> 1-hr gapless aggregation)
python src/data_engineering.py

# 2. Train the PyTorch LSTM model and export weights
python src/modeling.py

# 3. Compute Deep Feature Attributions & SHAP visualizations
python src/explainability.py

# 4. Generate Exploratory Data Analysis (EDA) charts
python src/eda_visualization.py
```

---

## 🧠 Model Architecture & Methodology

```text
[Input Window: 168 Hours × 15 Features]
       │
       ▼
[Layer 1: LSTM (hidden=128, dropout=0.2)]
       │
       ▼
[Layer 2: LSTM (hidden=128)]
       │
       ▼
[Linear Head: 128 -> 24]
       │
       ▼
[Forecast Output: 24 Hours Ahead]
```

* **Sequence Framing**: Lookback window of $168\text{ hours}$ (1 full week of historical electrical + meteorological readings).
* **Target Horizon**: Multi-step output of $24\text{ hours}$ ahead load curves.
* **Features Included**:
  * Electrical: `voltage_v`, `current_a`, `frequency_hz`, and autoregressive `energy_kwh` lag.
  * Meteorological: `temperature_c`, `humidity_pct`, `precipitation_mm`, `cloud_cover_pct`, `solar_radiation_wm2`, `wind_speed_ms`.
  * Calendar & Cyclical: `hour_sin`, `hour_cos`, `month_sin`, `month_cos`, `is_weekend`.

---

## 🌿 Carbon Footprint Factor

* **Grid Emission Baseline**: **$0.82\text{ kg CO}_2/\text{kWh}$** (Northern Regional Grid / Uttar Pradesh).
* **Reference**: *Central Electricity Authority (CEA) $\text{CO}_2$ Baseline Database for the Indian Power Sector*.

---

## 📜 License & Acknowledgements

* **Dataset**: Council on Energy, Environment and Water (CEEW) Smart Meter Data Mathura (2020) & Open-Meteo Historical Weather API.
* Developed for explainable energy forecasting and sustainable grid analytics.
