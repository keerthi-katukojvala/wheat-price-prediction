"""Paths and settings shared by the scripts. Paths are relative to the repository root."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW_PRICES_DIR = ROOT / "data" / "raw" / "agmarknet"
RAW_WEATHER_FILE = ROOT / "data" / "raw" / "weather" / "madhya_pradesh_daily_weather.xlsx"
MARKET_LOOKUP_FILE = ROOT / "data" / "reference" / "market_to_district.csv"

PROCESSED_DIR = ROOT / "data" / "processed"
MARKET_DAILY_FILE = PROCESSED_DIR / "wheat_market_daily_clean.csv"
DISTRICT_DAILY_FILE = PROCESSED_DIR / "wheat_district_daily_merged.csv"

RESULTS_DIR = ROOT / "results"

# Modal prices outside this range (Rs./quintal) are treated as data-entry errors.
PRICE_MIN, PRICE_MAX = 1200, 6000

# Time-based split: train on dates up to TRAIN_END, test on everything after.
TRAIN_END = "2022-10-31"

# District names in the weather workbook that differ from the names used here.
WEATHER_DISTRICT_RENAMES = {
    "Khargone (West Nimar)": "Khargone",
    "Khandwa(East Nimar)": "Khandwa",
    "Narsimhapur": "Narsinghpur",
}
