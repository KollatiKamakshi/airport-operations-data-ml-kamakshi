import time
import zipfile

import pandas as pd
import requests
import urllib3
from requests.exceptions import SSLError

from config import *

BTS_URL = ("https://transtats.bts.gov/PREZIP/"
           "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{y}_{m}.zip")
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
AIRPORTS_URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"


def _download(url, out, verify):
    with requests.get(url, stream=True, timeout=(30, 300), verify=verify) as r:
        r.raise_for_status()
        with open(out, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)


def download_bts():
    BTS_DIR.mkdir(parents=True, exist_ok=True)
    failed = []
    for m in range(1, 13):
        out = BTS_DIR / f"bts_{YEAR}_{m:02d}.zip"
        if out.exists() and zipfile.is_zipfile(out):
            continue
        url = BTS_URL.format(y=YEAR, m=m)
        print(f"Downloading BTS {YEAR}-{m:02d} (large file, please wait)...")
        try:
            try:
                _download(url, out, verify=True)
            except SSLError:
                print("  SSL certificate problem, retrying without the certificate check "
                      "(public government data)")
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                _download(url, out, verify=False)
            if not zipfile.is_zipfile(out):
                raise ValueError("downloaded file is not a valid zip")
            print("  OK")
        except Exception as e:
            out.unlink(missing_ok=True)
            failed.append(m)
            print(f"  FAILED ({e})")
    if failed:
        print(f"\nMonths still missing: {failed}")
        print("Run this script again, or download those months manually from the BTS TranStats site "
              f"and save them as {BTS_DIR}\\bts_{YEAR}_MM.zip")
    else:
        print("\nAll 12 BTS months are present.")


def get_airports():
    if AIRPORTS_FILE.exists():
        print("Airports file already exists, skipping.")
        return pd.read_csv(AIRPORTS_FILE)
    df = pd.read_csv(AIRPORTS_URL, low_memory=False)
    df = df[(df["type"] == "large_airport") & (df["iata_code"].isin(AIRPORTS))]
    sel = df[["iata_code", "name", "latitude_deg", "longitude_deg"]].drop_duplicates("iata_code")
    sel.to_csv(AIRPORTS_FILE, index=False)
    print("Airports found:", sorted(sel["iata_code"]))
    return sel


def download_weather(airports):
    if WEATHER_FILE.exists():
        print("Weather file already exists, skipping.")
        return
    frames = []
    for a in airports.itertuples():
        params = {
            "latitude": a.latitude_deg, "longitude": a.longitude_deg,
            "start_date": f"{YEAR}-01-01", "end_date": f"{YEAR}-12-31",
            "hourly": ",".join(WEATHER_VARS), "timezone": "auto",  # local time, same as BTS
        }
        for attempt in range(5):
            r = requests.get(ARCHIVE_URL, params=params, timeout=120)
            if r.status_code != 429:
                break
            time.sleep(30 * (attempt + 1))
        r.raise_for_status()
        h = pd.DataFrame(r.json()["hourly"])
        h["time"] = pd.to_datetime(h["time"])
        h.insert(0, "airport", a.iata_code)
        frames.append(h)
        print("Weather done:", a.iata_code)
        time.sleep(2)
    pd.concat(frames, ignore_index=True).to_parquet(WEATHER_FILE)


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    airports = get_airports()
    download_weather(airports)
    download_bts()
    print("Done.")