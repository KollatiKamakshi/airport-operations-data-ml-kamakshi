
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AIRPORTS = ROOT / "data" / "processed" / "airports.csv"
RUNWAYS = ROOT / "data" / "processed" / "runways.csv"

EXPECTED_CODES = {
    "ATL", "BOS", "DEN", "DFW", "JFK",
    "LAX", "MIA", "ORD", "SEA", "SFO",
}


def test_airports_have_expected_codes():
    airports = pd.read_csv(AIRPORTS)
    assert len(airports) == 10
    assert airports["iata_code"].nunique() == 10
    assert set(airports["iata_code"]) == EXPECTED_CODES


def test_airport_ids_are_unique():
    airports = pd.read_csv(AIRPORTS)
    assert airports["id"].notna().all()
    assert airports["id"].is_unique


def test_runways_reference_selected_airports():
    airports = pd.read_csv(AIRPORTS)
    runways = pd.read_csv(RUNWAYS)

    assert runways["id"].is_unique
    assert runways["airport_ref"].notna().all()
    assert set(runways["airport_ref"]).issubset(set(airports["id"]))


def test_airport_coordinates_are_valid():
    airports = pd.read_csv(AIRPORTS)

    assert airports["latitude_deg"].between(-90, 90).all()
    assert airports["longitude_deg"].between(-180, 180).all()
