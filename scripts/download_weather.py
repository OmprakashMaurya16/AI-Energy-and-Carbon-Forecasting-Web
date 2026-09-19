import json
import os
import urllib.request
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw")
WEATHER_PATH = os.path.join(RAW_DIR, "weather.csv")

# Coordinates: Bareilly (28.3670, 79.4304) / Mathura (27.4924, 77.6737)
LAT = 28.3670
LON = 79.4304
START_DATE = "2020-01-01"
END_DATE = "2020-12-31"

def fetch_weather():
    os.makedirs(RAW_DIR, exist_ok=True)
    
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={LAT}&longitude={LON}&"
        f"start_date={START_DATE}&end_date={END_DATE}&"
        f"hourly=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,cloud_cover,shortwave_radiation&"
        f"wind_speed_unit=ms&"
        f"timezone=Asia%2FKolkata"
    )
    
    print(f"Fetching 2020 weather data from Open-Meteo Archive API...")
    req = urllib.request.Request(url, headers={"User-Agent": "EnergyForecastApp/1.0"})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    
    hourly = data["hourly"]
    df = pd.DataFrame({
        "datetime": pd.to_datetime(hourly["time"]),
        "temperature_c": hourly["temperature_2m"],
        "humidity_pct": hourly["relative_humidity_2m"],
        "precipitation_mm": hourly["precipitation"],
        "wind_speed_ms": hourly["wind_speed_10m"],
        "cloud_cover_pct": hourly["cloud_cover"],
        "solar_radiation_wm2": hourly["shortwave_radiation"],
    })
    
    # Backup previous weather file if it exists and is different
    if os.path.exists(WEATHER_PATH):
        backup_path = os.path.join(RAW_DIR, "weather_old_backup.csv")
        if not os.path.exists(backup_path):
            os.rename(WEATHER_PATH, backup_path)
            print(f"Backed up old weather.csv -> {backup_path}")
            
    df.to_csv(WEATHER_PATH, index=False)
    print(f"[OK] Saved 2020 weather data ({len(df)} hourly rows) -> {WEATHER_PATH}")

if __name__ == "__main__":
    fetch_weather()
