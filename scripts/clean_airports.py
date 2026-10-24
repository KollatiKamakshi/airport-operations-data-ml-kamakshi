
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

SELECTED_FILE = DATA_DIR / "airports_selected.csv"
RAW_AIRPORTS_FILE = RAW_DIR / "airports.csv"
RAW_RUNWAYS_FILE = RAW_DIR / "runways.csv"


def clean_airports():
    selected = pd.read_csv(SELECTED_FILE)
    airports = pd.read_csv(RAW_AIRPORTS_FILE, low_memory=False)

    # Normalize airport codes before matching.
    selected["iata_code"] = (
        selected["iata_code"].astype("string").str.strip().str.upper()
    )
    airports["iata_code"] = (
        airports["iata_code"].astype("string").str.strip().str.upper()
    )

    # Keep only the target airports, and only major airports.
    cleaned = airports[
        airports["iata_code"].isin(selected["iata_code"])
        & airports["type"].eq("large_airport")
    ].copy()

    # One airport record per IATA code.
    cleaned = cleaned.drop_duplicates(subset=["iata_code"], keep="first")

    columns = [
        "id",
        "ident",
        "iata_code",
        "icao_code",
        "name",
        "type",
        "latitude_deg",
        "longitude_deg",
        "elevation_ft",
        "iso_country",
        "iso_region",
        "municipality",
        "scheduled_service",
    ]
    cleaned = cleaned[columns]

    # Reject invalid or missing identity and coordinate values.
    cleaned = cleaned.dropna(
        subset=["id", "ident", "iata_code", "name",
                "latitude_deg", "longitude_deg"]
    )
    cleaned = cleaned[
        cleaned["latitude_deg"].between(-90, 90)
        & cleaned["longitude_deg"].between(-180, 180)
    ]

    # Verify that every selected airport was found.
    missing_codes = sorted(
        set(selected["iata_code"]) - set(cleaned["iata_code"])
    )
    if missing_codes:
        raise ValueError(f"Selected airports missing from cleaned data: {missing_codes}")

    # Keep the selected-airport order.
    cleaned = (
        selected[["iata_code"]]
        .merge(cleaned, on="iata_code", how="left", validate="one_to_one")
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output = PROCESSED_DIR / "airports.csv"
    cleaned.to_csv(output, index=False)

    print(f"Saved {len(cleaned)} airports to {output}")
    print(cleaned[["iata_code", "name", "ident"]].to_string(index=False))


def clean_runways():
    airports = pd.read_csv(PROCESSED_DIR / "airports.csv")
    runways = pd.read_csv(RAW_RUNWAYS_FILE, low_memory=False)

    # Match runways using OurAirports' stable airport reference ID.
    runways["airport_ref"] = pd.to_numeric(
        runways["airport_ref"], errors="coerce"
    )
    airport_ids = set(pd.to_numeric(airports["id"]).astype("int64"))

    cleaned = runways[runways["airport_ref"].isin(airport_ids)].copy()

    # Keep closed runways in the data for traceability; flag them explicitly.
    cleaned["closed"] = pd.to_numeric(cleaned["closed"], errors="coerce")
    cleaned["length_ft"] = pd.to_numeric(
        cleaned["length_ft"], errors="coerce"
    )
    cleaned["width_ft"] = pd.to_numeric(
        cleaned["width_ft"], errors="coerce"
    )

    # Remove duplicate runway records and retain useful operational fields.
    cleaned = cleaned.drop_duplicates(subset=["id"])
    columns = [
        "id",
        "airport_ref",
        "airport_ident",
        "length_ft",
        "width_ft",
        "surface",
        "lighted",
        "closed",
        "le_ident",
        "le_heading_degT",
        "he_ident",
        "he_heading_degT",
    ]
    cleaned = cleaned[columns]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output = PROCESSED_DIR / "runways.csv"
    cleaned.to_csv(output, index=False)

    print(f"\nSaved {len(cleaned)} runway records to {output}")
    print("Runways per airport:")
    print(
        cleaned.groupby("airport_ident")
        .size()
        .sort_index()
        .to_string()
    )


if __name__ == "__main__":
    clean_airports()
    clean_runways()
