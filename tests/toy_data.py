"""
TEST-ONLY toy data. This is FAKE data used to test code before the real
dataset exists. Never use it for results, charts in the report, or metrics.

Schema follows PRD Section 5.4 (data/flights_model.parquet).
"""

import numpy as np
import pandas as pd

# Weather columns are listed here explicitly so this file works on its own.
# They must match WEATHER_FEATURES in config.py (the checker will compare them).
_WEATHER_VARS = ["temperature_2m", "precipitation", "cloud_cover",
                 "wind_speed_10m", "wind_gusts_10m", "weather_code"]

_AIRPORTS = ["ATL", "ORD", "DFW", "DEN", "JFK", "LAX", "SFO", "SEA", "MIA", "BOS"]
_AIRLINES = ["AA", "DL", "UA", "WN", "B6", "AS"]


def make_toy_flights(n_rows=2000, seed=42):
    """Return a clean, fake DataFrame that follows the Section 5.4 schema."""
    rng = np.random.default_rng(seed)

    # Random dates across the whole of 2023
    days = rng.integers(0, 365, size=n_rows)
    flight_date = pd.Timestamp("2023-01-01") + pd.to_timedelta(days, unit="D")

    # Pick origin and dest from the 10 airports, making sure they differ
    origin = rng.choice(_AIRPORTS, size=n_rows)
    dest = rng.choice(_AIRPORTS, size=n_rows)
    same = origin == dest
    while same.any():
        dest[same] = rng.choice(_AIRPORTS, size=same.sum())
        same = origin == dest

    df = pd.DataFrame({
        "flight_date": flight_date,
        "tail_number": [f"N{x}" for x in rng.integers(100, 999, size=n_rows)],
        "airline": rng.choice(_AIRLINES, size=n_rows),
        "origin": origin,
        "dest": dest,
        "dep_hour": rng.integers(5, 23, size=n_rows),
        "distance": rng.uniform(200, 2700, size=n_rows).round(0),
    })

    df["route"] = df["origin"] + "_" + df["dest"]
    df["month"] = df["flight_date"].dt.month
    df["day_of_week"] = df["flight_date"].dt.dayofweek + 1  # BTS: 1=Mon ... 7=Sun
    df["is_holiday"] = rng.choice([0, 1], size=n_rows, p=[0.97, 0.03])
    df["sched_duration"] = (df["distance"] / 7.5 + 30).round(0)  # rough minutes
    df["origin_dep_count"] = rng.integers(1, 60, size=n_rows)

    # Fake weather for origin (o_) and destination (d_)
    for prefix in ("o_", "d_"):
        df[prefix + "temperature_2m"] = rng.normal(15, 10, size=n_rows)
        df[prefix + "precipitation"] = rng.exponential(0.3, size=n_rows)
        df[prefix + "cloud_cover"] = rng.uniform(0, 100, size=n_rows)
        df[prefix + "wind_speed_10m"] = rng.uniform(0, 30, size=n_rows)
        df[prefix + "wind_gusts_10m"] = df[prefix + "wind_speed_10m"] + rng.uniform(0, 15, size=n_rows)
        df[prefix + "weather_code"] = rng.choice([0, 1, 3, 61, 71, 95], size=n_rows)

    # Labels: delay_min is clipped to 0..300; delayed = delay above 15 minutes
    raw_delay = rng.exponential(12, size=n_rows) - 5
    df["delay_min"] = np.clip(raw_delay, 0, 300).round(1)
    df["delayed"] = (df["delay_min"] > 15).astype(int)

    return df


def make_broken_versions(clean_df):
    """
    Return a dict of {name: broken DataFrame}. Each one has ONE deliberate
    problem so we can check that 04_check_dataset.py catches it.
    """
    broken = {}

    # 1. A required column is missing
    broken["missing_column"] = clean_df.drop(columns=["dep_hour"])

    # 2. A leaky column was added (departure delay)
    leaky = clean_df.copy()
    leaky["dep_delay"] = np.random.default_rng(0).normal(10, 20, size=len(leaky))
    broken["leaky_column"] = leaky

    # 3. An airport that is not in our 10-airport list
    bad_airport = clean_df.copy()
    bad_airport.loc[bad_airport.index[0], "origin"] = "XXX"
    broken["unknown_airport"] = bad_airport

    # 4. Impossible hour
    bad_hour = clean_df.copy()
    bad_hour.loc[bad_hour.index[0], "dep_hour"] = 27
    broken["bad_hour"] = bad_hour

    # 5. 'delayed' label disagrees with 'delay_min'
    bad_label = clean_df.copy()
    bad_label.loc[bad_label.index[0], "delayed"] = 1 - bad_label.loc[bad_label.index[0], "delayed"]
    broken["label_mismatch"] = bad_label

    return broken