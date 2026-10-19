"""Create a synthetic daily weather + solar output dataset (solar_data.csv).

Each row is one day for a 5 kW rooftop system at ~20 degrees latitude.
Output (kWh) comes from sun position, cloud cover, and a heat penalty,
plus random noise. A few weather readings are blanked out on purpose so
the data-preparation step has real missing values to deal with.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 1500
KWP, LAT = 5.0, 20.0

dates = pd.date_range("2022-01-01", periods=N, freq="D")
doy = dates.dayofyear.values
month = dates.month.values

# Seasonal weather: a monsoon-like cloudy season in the middle of the year
season = np.sin(2 * np.pi * (doy - 80) / 365)           # -1..1
cloud = np.clip(45 + 25 * season + rng.normal(0, 28, N), 0, 100)
temp_max = 30 + 5 * np.sin(2 * np.pi * (doy - 100) / 365) - 0.03 * cloud + rng.normal(0, 2, N)
temp_min = temp_max - rng.uniform(5, 11, N)
humidity = np.clip(40 + 0.45 * cloud + rng.normal(0, 8, N), 15, 100)
rain_mm = np.where(cloud > 65, rng.gamma(2.0, 4.0, N) * (cloud - 60) / 40, 0.0)
wind = np.clip(rng.normal(12, 5, N) - 0.04 * cloud, 0, None)

# Clear-sky daily energy from sun geometry
dec = np.radians(23.45 * np.sin(2 * np.pi * (284 + doy) / 365))
phi = np.radians(LAT)
day_len = 24 / np.pi * np.arccos(np.clip(-np.tan(phi) * np.tan(dec), -1, 1))
peak_sun_hours = 6.0 * (np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec)) ** 0.5 * day_len / 12
cloud_factor = 1 - 0.75 * (cloud / 100) ** 3.4
heat = 1 - 0.004 * np.clip(temp_max - 25 + 8, 0, None)
kwh = KWP * peak_sun_hours * 0.86 * cloud_factor * heat
kwh = np.clip(kwh * rng.normal(1, 0.06, N), 0.2, None)

df = pd.DataFrame({
    "date": dates, "month": month, "cloud_cover_pct": cloud.round(1),
    "temp_max_c": temp_max.round(1), "temp_min_c": temp_min.round(1),
    "humidity_pct": humidity.round(1), "rain_mm": rain_mm.round(1),
    "wind_kmh": wind.round(1), "day_length_h": day_len.round(2),
    "energy_kwh": kwh.round(2),
})

# Target: 3 classes using quantile cut-offs so classes are roughly balanced
lo, hi = df["energy_kwh"].quantile([1 / 3, 2 / 3])
df["output_level"] = pd.cut(df["energy_kwh"], [-np.inf, lo, hi, np.inf],
                            labels=["Low", "Medium", "High"])

# Blank out ~3% of some readings to mimic faulty sensors
for col in ["cloud_cover_pct", "humidity_pct", "wind_kmh", "temp_max_c"]:
    df.loc[rng.random(N) < 0.03, col] = np.nan

df.to_csv("solar_data.csv", index=False)
print(f"Saved solar_data.csv with {len(df)} rows. Class cut-offs: Low < {lo:.1f} < Medium < {hi:.1f} < High (kWh)")
