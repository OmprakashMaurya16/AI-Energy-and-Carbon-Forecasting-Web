import pandas as pd
from pathlib import Path

root = Path(__file__).resolve().parents[1]
raw = root / "data" / "raw" / "Weather.csv"
out = root / "data" / "raw" / "weather.csv"

if not raw.exists():
    raise SystemExit(f"Missing: {raw}")

df = pd.read_csv(raw, skiprows=2, parse_dates=["time"]) 

df = df.rename(columns={
    "time": "datetime",
    "temperature_2m (°C)": "temperature_c",
    "relativehumidity_2m (%)": "humidity_pct",
    "precipitation (mm)": "precipitation_mm",
    "windspeed_10m (km/h)": "wind_speed_kmh",
    "cloudcover (%)": "cloud_cover_pct",
    "shortwave_radiation (W/m²)": "solar_radiation_wm2",
})

if "wind_speed_kmh" in df.columns:
    df["wind_speed_ms"] = df["wind_speed_kmh"] / 3.6
    df = df.drop(columns=["wind_speed_kmh"])

out_dir = out.parent
out_dir.mkdir(parents=True, exist_ok=True)

df.to_csv(out, index=False)
print(f"Written normalized weather CSV → {out}")
