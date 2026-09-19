import json
import os
import sys
import urllib.request
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DATA_DIR = os.path.join(_ROOT, "data", "processed")
POSOCO_CSV_PATH = os.path.join(DATA_DIR, "posoco_states.csv")

# Official Central Electricity Authority (CEA) CO2 Baseline Database for Indian Power Sector (Version 18/19)
# Values in kg CO2 / kWh (or tCO2 / MWh)
STATE_EMISSION_FACTORS = {
    "Uttar Pradesh": 0.82,
    "Maharashtra": 0.85,
    "Gujarat": 0.84,
    "Tamil Nadu": 0.72,
    "Karnataka": 0.73,
    "Delhi": 0.82,
    "Rajasthan": 0.83,
    "Punjab": 0.81,
    "Haryana": 0.82,
    "Madhya Pradesh": 0.86,
    "West Bengal": 0.95,
    "Bihar": 0.92,
    "Andhra Pradesh": 0.75,
    "Telangana": 0.76,
    "Kerala": 0.68,
    "Odisha": 0.96,
    "Chhattisgarh": 0.93,
    "Uttarakhand": 0.65,
    "Himachal Pradesh": 0.45,
    "Goa": 0.84,
    "Assam": 0.78,
    "Jharkhand": 0.94,
    "National Average": 0.82,
}

STATE_REGIONS = {
    "Uttar Pradesh": "Northern",
    "Delhi": "Northern",
    "Rajasthan": "Northern",
    "Punjab": "Northern",
    "Haryana": "Northern",
    "Uttarakhand": "Northern",
    "Himachal Pradesh": "Northern",
    "Maharashtra": "Western",
    "Gujarat": "Western",
    "Madhya Pradesh": "Western",
    "Chhattisgarh": "Western",
    "Goa": "Western",
    "Tamil Nadu": "Southern",
    "Karnataka": "Southern",
    "Andhra Pradesh": "Southern",
    "Telangana": "Southern",
    "Kerala": "Southern",
    "West Bengal": "Eastern",
    "Bihar": "Eastern",
    "Odisha": "Eastern",
    "Jharkhand": "Eastern",
    "Assam": "North-Eastern",
}

# Mapping short column codes to proper state names
COLUMN_MAPPING = {
    "UP": "Uttar Pradesh",
    "MP": "Madhya Pradesh",
    "HP": "Himachal Pradesh",
    "J&K": "Jammu & Kashmir",
    "DNH": "Dadra and Nagar Haveli",
    "Pondy": "Puducherry",
}

POSOCO_RAW_URL = "https://raw.githubusercontent.com/twinkle0705/Power-Consumption-Dataset/master/dataset_tk.csv"


def download_or_load_posoco_data() -> pd.DataFrame:
    """
    Downloads POSOCO official state-level power consumption dataset if not cached,
    cleans the date index, renames columns, and returns a clean DataFrame (in Mega Units - MU).
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(POSOCO_CSV_PATH):
        df = pd.read_csv(POSOCO_CSV_PATH, parse_dates=["datetime"], index_col="datetime")
        return df

    print(f"Downloading official POSOCO multi-state dataset from GitHub repository...")
    req = urllib.request.Request(POSOCO_RAW_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        raw_df = pd.read_csv(resp)

    # First column contains datetime
    date_col = raw_df.columns[0]
    raw_df["datetime"] = pd.to_datetime(raw_df[date_col], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    raw_df = raw_df.dropna(subset=["datetime"]).sort_values("datetime").set_index("datetime")
    raw_df = raw_df.drop(columns=[date_col])

    # Rename short codes to standardized state names
    rename_dict = {}
    for col in raw_df.columns:
        clean_name = COLUMN_MAPPING.get(col, col.strip())
        rename_dict[col] = clean_name
    raw_df = raw_df.rename(columns=rename_dict)

    # Ensure numeric & handle missing
    raw_df = raw_df.apply(pd.to_numeric, errors="coerce")
    raw_df = raw_df.interpolate(method="time").ffill().bfill()

    raw_df.to_csv(POSOCO_CSV_PATH)
    print(f"[OK] Saved POSOCO multi-state dataset to {POSOCO_CSV_PATH} ({raw_df.shape[0]} days, {raw_df.shape[1]} states)")
    return raw_df


def get_available_states() -> list:
    df = download_or_load_posoco_data()
    return sorted([c for c in df.columns if c in STATE_EMISSION_FACTORS])


def get_state_summary_statistics() -> pd.DataFrame:
    """Returns a summary table for all states with demand, emission factors, and volatility."""
    df = download_or_load_posoco_data()
    states = get_available_states()
    
    rows = []
    for s in states:
        series = df[s]
        ef = STATE_EMISSION_FACTORS.get(s, 0.82)
        region = STATE_REGIONS.get(s, "Other")
        mean_mu = series.mean()
        peak_mu = series.max()
        min_mu = series.min()
        std_mu = series.std()
        cv = (std_mu / mean_mu) * 100 if mean_mu > 0 else 0
        
        # Total daily emissions in Metric Tonnes CO2: (MU * 1,000,000 kWh * ef kg) / 1000 kg = MU * 1000 * ef
        daily_co2_tonnes = mean_mu * 1000 * ef
        
        rows.append({
            "State": s,
            "Region": region,
            "Avg Daily Demand (MU)": round(mean_mu, 2),
            "Peak Demand (MU)": round(peak_mu, 2),
            "Min Demand (MU)": round(min_mu, 2),
            "Volatility (CV %)": round(cv, 2),
            "CEA Grid Emission (kg CO2/kWh)": ef,
            "Avg Daily CO2 (Tonnes)": round(daily_co2_tonnes, 1),
        })
        
    res_df = pd.DataFrame(rows).sort_values("Avg Daily Demand (MU)", ascending=False).reset_index(drop=True)
    return res_df


STATE_WEATHER_PATH = os.path.join(DATA_DIR, "state_weather.csv")

STATE_CAPITALS = {
    "Maharashtra": {"city": "Mumbai", "lat": 18.9388, "lon": 72.8354},
    "Gujarat": {"city": "Ahmedabad", "lat": 23.0225, "lon": 72.5714},
    "Uttar Pradesh": {"city": "Lucknow", "lat": 26.8467, "lon": 80.9462},
    "Tamil Nadu": {"city": "Chennai", "lat": 13.0827, "lon": 80.2707},
    "Karnataka": {"city": "Bengaluru", "lat": 12.9716, "lon": 77.5946},
    "Delhi": {"city": "New Delhi", "lat": 28.6139, "lon": 77.2090},
    "Rajasthan": {"city": "Jaipur", "lat": 26.9124, "lon": 75.7873},
    "Punjab": {"city": "Chandigarh", "lat": 30.7333, "lon": 76.7794},
    "West Bengal": {"city": "Kolkata", "lat": 22.5726, "lon": 88.3639},
    "Kerala": {"city": "Thiruvananthapuram", "lat": 8.5241, "lon": 76.9366},
    "Madhya Pradesh": {"city": "Bhopal", "lat": 23.2599, "lon": 77.4126},
    "Bihar": {"city": "Patna", "lat": 25.5941, "lon": 85.1376},
    "Odisha": {"city": "Bhubaneswar", "lat": 20.2961, "lon": 85.8245},
}


def download_or_load_state_weather() -> pd.DataFrame:
    """
    Downloads or loads historical 2019-2020 daily weather (temperature, humidity)
    for state capital hubs from Open-Meteo Historical Archive API.
    """
    if os.path.exists(STATE_WEATHER_PATH):
        return pd.read_csv(STATE_WEATHER_PATH, parse_dates=["datetime"], index_col="datetime")

    print("Fetching historical 2019-2020 weather for state capital hubs from Open-Meteo API...")
    posoco_df = download_or_load_posoco_data()
    start_str = posoco_df.index.min().strftime("%Y-%m-%d")
    end_str = posoco_df.index.max().strftime("%Y-%m-%d")

    weather_dfs = []
    for state, info in STATE_CAPITALS.items():
        try:
            url = (
                f"https://archive-api.open-meteo.com/v1/archive?"
                f"latitude={info['lat']}&longitude={info['lon']}&"
                f"start_date={start_str}&end_date={end_str}&"
                f"daily=temperature_2m_mean,temperature_2m_max,relative_humidity_2m_mean&"
                f"timezone=Asia%2FKolkata"
            )
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            daily = data["daily"]
            st_df = pd.DataFrame({
                "datetime": pd.to_datetime(daily["time"]),
                f"{state}_temp_mean_c": daily["temperature_2m_mean"],
                f"{state}_temp_max_c": daily["temperature_2m_max"],
                f"{state}_humidity_pct": daily["relative_humidity_2m_mean"],
            }).set_index("datetime")
            weather_dfs.append(st_df)
            print(f"  [OK] Fetched weather for {state} ({info['city']})")
        except Exception as e:
            print(f"  [WARN] Could not fetch weather for {state}: {e}")

    if weather_dfs:
        merged_weather = weather_dfs[0]
        for wdf in weather_dfs[1:]:
            merged_weather = merged_weather.join(wdf, how="outer")
        merged_weather = merged_weather.interpolate(method="time").ffill().bfill()
        merged_weather.to_csv(STATE_WEATHER_PATH)
        print(f"[OK] Saved multi-state weather dataset -> {STATE_WEATHER_PATH}")
        return merged_weather
    return pd.DataFrame()


def get_state_weather_merged(state_name: str) -> pd.DataFrame:
    """Returns a merged DataFrame of daily power demand and Open-Meteo weather for the specified state."""
    posoco_df = download_or_load_posoco_data()
    weather_df = download_or_load_state_weather()

    if state_name not in posoco_df.columns:
        return pd.DataFrame()

    res = pd.DataFrame({"demand_mu": posoco_df[state_name]})
    temp_col = f"{state_name}_temp_mean_c"
    max_col = f"{state_name}_temp_max_c"
    hum_col = f"{state_name}_humidity_pct"

    if temp_col in weather_df.columns:
        res["temperature_mean_c"] = weather_df[temp_col]
        res["temperature_max_c"] = weather_df[max_col]
        res["humidity_pct"] = weather_df[hum_col]
        res["cooling_degree_days"] = (res["temperature_mean_c"] - 24.0).clip(lower=0)
    return res.dropna()


if __name__ == "__main__":
    df = download_or_load_posoco_data()
    print("Multi-state shape:", df.shape)
    wdf = download_or_load_state_weather()
    print("State weather shape:", wdf.shape)
    summary = get_state_summary_statistics()
    print("\nTop 10 States by Power Demand:")
    print(summary.head(10).to_string())
