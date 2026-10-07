"""
Step 1: build the modelling dataset.

Reads the raw Agmarknet wheat price files and the district weather workbook,
cleans them, and writes two files to data/processed/:

  wheat_market_daily_clean.csv     one row per market, variety and day
  wheat_district_daily_merged.csv  one row per district and day, with weather
                                   and engineered features (used for modelling)

Run from anywhere:  python src/build_dataset.py
"""
import io

import numpy as np
import pandas as pd

import config as cfg


def load_prices():
    """Read every monthly Agmarknet export (HTML tables) and clean it."""
    files = sorted(cfg.RAW_PRICES_DIR.glob("wheat_*.html"))
    if not files:
        raise FileNotFoundError(f"No price files found in {cfg.RAW_PRICES_DIR}")

    frames = [pd.read_html(io.StringIO(f.read_text(encoding="latin-1")))[0] for f in files]
    df = pd.concat(frames, ignore_index=True)
    df.columns = ["market", "date", "arrivals_tonnes", "variety",
                  "min_price", "max_price", "modal_price"]
    log = {"price files read": len(files), "raw rows": len(df)}

    # The market name is printed only on the first row of each block.
    df["market"] = df["market"].ffill()
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
    for col in ["arrivals_tonnes", "min_price", "max_price", "modal_price"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["date"])                       # undated per-variety summary rows
    log["dropped: no date"] = before - len(df)

    before = len(df)
    df = df.dropna(subset=["modal_price"])
    df = df[df["modal_price"].between(cfg.PRICE_MIN, cfg.PRICE_MAX)]
    log[f"dropped: modal price missing or outside {cfg.PRICE_MIN}-{cfg.PRICE_MAX}"] = before - len(df)

    before = len(df)
    df = df.drop_duplicates()
    log["dropped: exact duplicates"] = before - len(df)

    log["clean market-level rows"] = len(df)
    return df.reset_index(drop=True), log


def add_district(prices):
    """Attach the district of each market using the reference lookup."""
    lookup = pd.read_csv(cfg.MARKET_LOOKUP_FILE)[["market", "district"]]
    out = prices.merge(lookup, on="market", how="left", validate="many_to_one")
    unmapped = sorted(out.loc[out["district"].isna(), "market"].unique())
    if unmapped:
        raise ValueError(f"Markets missing from {cfg.MARKET_LOOKUP_FILE.name}: {unmapped}")
    return out[["district", "market", "date", "variety", "arrivals_tonnes",
                "min_price", "max_price", "modal_price"]]


def aggregate_to_district_day(prices):
    """Collapse market rows to one row per district and day (arrival-weighted)."""
    df = prices.copy()
    df["weight"] = df["arrivals_tonnes"].fillna(0).clip(lower=0)
    is_sharbati = df["variety"].str.contains("sharbati", case=False, na=False)
    df["sharbati_weight"] = df["weight"] * is_sharbati

    def summarise(group):
        total = group["weight"].sum()
        if total > 0:
            price = np.average(group["modal_price"], weights=group["weight"])
            sharbati_share = group["sharbati_weight"].sum() / total
        else:                                            # no arrivals recorded: simple mean
            price, sharbati_share = group["modal_price"].mean(), 0.0
        return pd.Series({
            "modal_price": price,
            "arrivals_tonnes": total,
            "sharbati_share": sharbati_share,
            "n_markets": group["market"].nunique(),
        })

    return df.groupby(["district", "date"]).apply(summarise, include_groups=False).reset_index()


def load_weather_features():
    """Rolling 30- and 90-day weather per district, using days before the price date only."""
    sheets = pd.read_excel(cfg.RAW_WEATHER_FILE, sheet_name=["rain", "tmax", "tmin"])
    features = []
    for variable, sheet in sheets.items():
        daily = (sheet.rename(columns=cfg.WEATHER_DISTRICT_RENAMES)
                      .set_index("DateTime").sort_index().loc["2021-06-01":])
        previous = daily.shift(1)                        # exclude the price date itself
        for window in (30, 90):
            rolled = previous.rolling(window)
            rolled = rolled.sum() if variable == "rain" else rolled.mean()
            features.append(rolled.stack().rename(f"{variable}_{window}d"))
    weather = pd.concat(features, axis=1).reset_index()
    return weather.rename(columns={"DateTime": "date", "level_1": "district"})


def add_model_features(df):
    """Lagged prices, season and trend features."""
    df = df.sort_values(["district", "date"]).reset_index(drop=True)
    by_district = df.groupby("district")["modal_price"]
    df["price_lag1"] = by_district.shift(1)              # previous trading day
    df["price_prev5_mean"] = by_district.transform(
        lambda s: s.shift(1).rolling(5, min_periods=3).mean())
    df["days_since_prev"] = df.groupby("district")["date"].diff().dt.days
    df["log_arrivals"] = np.log1p(df["arrivals_tonnes"])
    df["month"] = df["date"].dt.month
    df["harvest_season"] = df["month"].isin([3, 4, 5]).astype(int)   # rabi wheat arrivals
    df["days_from_start"] = (df["date"] - df["date"].min()).dt.days
    return df


def main():
    prices, log = load_prices()
    prices = add_district(prices)

    district_daily = aggregate_to_district_day(prices)
    weather = load_weather_features()
    merged = district_daily.merge(weather, on=["date", "district"], how="left",
                                  validate="one_to_one")
    log["district-day rows"] = len(merged)
    log["districts"] = merged["district"].nunique()
    log["rows without a weather match"] = int(merged["rain_30d"].isna().sum())
    merged = add_model_features(merged)

    cfg.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    prices.to_csv(cfg.MARKET_DAILY_FILE, index=False)
    merged.to_csv(cfg.DISTRICT_DAILY_FILE, index=False, float_format="%.4f")

    print("Dataset build summary")
    for step, count in log.items():
        print(f"  {step}: {count:,}")
    print(f"\nSaved {cfg.MARKET_DAILY_FILE.relative_to(cfg.ROOT)}")
    print(f"Saved {cfg.DISTRICT_DAILY_FILE.relative_to(cfg.ROOT)}")


if __name__ == "__main__":
    main()
