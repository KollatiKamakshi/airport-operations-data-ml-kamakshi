"""
04_check_dataset.py  (owner: Kamakshi)

Quality gate for the flight dataset. Prints PASS or FAIL for each check.

Usage (from the flight-delay folder):
    python 04_check_dataset.py                  # checks data/flights_model.parquet
    python 04_check_dataset.py path\to\file.parquet
"""

import sys
import pandas as pd
from pandas.api import types as pdt

from config import AIRPORTS, DATASET_FILE, WEATHER_FEATURES

# ---------------------------------------------------------------------
# Required columns from PRD Section 5.4, with the kind of type we expect.
# kinds: "datetime", "string", "int", "float"
# NOTE: history features (airline_delay_rate etc.) are NOT here on purpose.
# They are added at training time, not stored in the file.
# ---------------------------------------------------------------------
REQUIRED_COLUMNS = {
    "flight_date": "datetime",
    "tail_number": "string",
    "route": "string",
    "airline": "string",
    "origin": "string",
    "dest": "string",
    "month": "int",
    "day_of_week": "int",
    "dep_hour": "int",
    "is_holiday": "int",
    "distance": "float",
    "sched_duration": "float",
    "origin_dep_count": "int",
    "delayed": "int",
    "delay_min": "float",
}
for _col in WEATHER_FEATURES:
    # weather_code is a category number but stored as float in the PRD
    REQUIRED_COLUMNS[_col] = "float"

# ---------------------------------------------------------------------
# Leaky columns from PRD Section 5.10. We check for these words inside
# column names (lower case). Add more words here if the team finds new ones.
# ---------------------------------------------------------------------
LEAKY_KEYWORDS = [
    "dep_delay", "departure_delay",
    "taxi_out", "taxi_in", "taxiout", "taxiin",
    "wheels_off", "wheels_on", "wheelsoff", "wheelson",
    "arr_time", "actual_arr", "actual_elapsed", "air_time",
    "cancel",
    "carrier_delay", "weather_delay", "nas_delay",
    "security_delay", "late_aircraft",
]


# ---------------------------------------------------------------------
# Small helper so every check reports in the same format
# ---------------------------------------------------------------------
def make_result(name, passed, message):
    return {"name": name, "passed": bool(passed), "message": message}


def type_matches(series, kind):
    """True if a pandas column has the kind of type we expect."""
    if kind == "datetime":
        return pdt.is_datetime64_any_dtype(series)
    if kind == "string":
        return pdt.is_string_dtype(series) or pdt.is_object_dtype(series)
    if kind == "int":
        return pdt.is_integer_dtype(series)
    if kind == "float":
        return pdt.is_numeric_dtype(series)  # int or float both acceptable
    return False


# ---------------------------------------------------------------------
# Check 1: schema
# ---------------------------------------------------------------------
def check_schema(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return make_result("Schema: required columns", False,
                           f"missing columns: {missing}")

    wrong_type = [c for c, kind in REQUIRED_COLUMNS.items()
                  if not type_matches(df[c], kind)]
    if wrong_type:
        return make_result("Schema: column types", False,
                           f"unexpected type in columns: {wrong_type}")

    return make_result("Schema", True,
                       f"all {len(REQUIRED_COLUMNS)} required columns present with sensible types")


# ---------------------------------------------------------------------
# Check 2: leakage
# ---------------------------------------------------------------------
def check_leakage(df):
    found = []
    for col in df.columns:
        lowered = col.lower()
        if any(word in lowered for word in LEAKY_KEYWORDS):
            found.append(col)
    if found:
        return make_result("Leakage", False, f"leaky columns found: {found}")
    return make_result("Leakage", True, "no leaky column names found")


# ---------------------------------------------------------------------
# Check 3: missing values (a report, with a FAIL if the key columns have gaps)
# ---------------------------------------------------------------------
def check_missing(df):
    counts = df.isna().sum()
    counts = counts[counts > 0].sort_values(ascending=False)

    if counts.empty:
        return make_result("Missing values", True, "no missing values in any column")

    lines = [f"{col}: {n} ({n / len(df):.1%})" for col, n in counts.items()]
    # Labels must never be missing. Weather may have a few gaps (PRD: "almost never").
    label_gaps = [c for c in ("delayed", "delay_min") if c in counts.index]
    passed = not label_gaps
    prefix = "labels have missing values! " if label_gaps else "report only: "
    return make_result("Missing values", passed, prefix + "; ".join(lines))


# ---------------------------------------------------------------------
# Run everything and print a clear report
# ---------------------------------------------------------------------
def run_checks(df):
    return [check_schema(df), check_leakage(df), check_missing(df)]


def print_report(results):
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"[{status}] {r['name']}: {r['message']}")
    overall = all(r["passed"] for r in results)
    print("-" * 60)
    print("OVERALL:", "PASS" if overall else "FAIL")
    return overall


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else DATASET_FILE
    print(f"Checking: {path}")
    data = pd.read_parquet(path)
    print(f"Rows: {len(data):,}   Columns: {data.shape[1]}")
    print("-" * 60)
    ok = print_report(run_checks(data))
    sys.exit(0 if ok else 1)