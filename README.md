# Wheat Price Prediction — Madhya Pradesh

Predicting daily wheat mandi prices across the districts of Madhya Pradesh, India, from
market arrivals, wheat variety, season and district weather, using linear, ridge and lasso
regression.

The project started during an internship in India. This repository contains the raw data,
a reproducible cleaning and merging pipeline, the resulting dataset, and the model comparison.

## What is in the data

| Source | Contents | Period |
|---|---|---|
| [Agmarknet](https://agmarknet.gov.in) | Daily wheat prices and arrivals for 252 mandis in Madhya Pradesh | Jan 2022 – Jan 2023 |
| District weather workbook (extracted in QGIS) | Daily rainfall, maximum and minimum temperature for 50 districts | 1985 – 2023 |

I extracted the weather series (rainfall, maximum and minimum temperature) in QGIS from
gridded data, sampling one coordinate per district.

After cleaning: **54,715** market-level price records, aggregated to **16,851** district-day
rows covering all 50 districts.

## Repository layout

```
data/
  raw/agmarknet/         13 monthly Agmarknet exports (HTML tables), wheat_YYYY_MM.html
  raw/weather/           madhya_pradesh_daily_weather.xlsx (sheets: rain, tmax, tmin)
  reference/             market_to_district.csv  (mandi -> district lookup)
  processed/             cleaned outputs written by build_dataset.py
src/
  config.py              paths and settings
  build_dataset.py       step 1: clean, map, aggregate, merge, add features
  train_models.py        step 2: linear / ridge / lasso and baselines
notebooks/
  wheat_price_prediction.ipynb   the whole pipeline in one notebook, with outputs and charts
results/                 model_results.csv, model_coefficients.csv
```

## How to run

Requires Python 3.9+.

```bash
pip install -r requirements.txt
python src/build_dataset.py
python src/train_models.py
```

Or open `notebooks/wheat_price_prediction.ipynb` in Jupyter and run all cells; it performs the
same steps in one place.

The processed data and results are already committed, so the second script can be run on
its own.

## Pipeline

1. **Parse prices.** The Agmarknet exports are HTML tables. The market name appears only on
   the first row of each block, so it is filled down; dates and prices are converted to
   proper types.
2. **Clean.** Drop 4,244 undated per-variety summary rows and 8 rows whose modal price is
   outside 1,200–6,000 Rs./quintal (treated as entry errors).
3. **Map mandis to districts** with `data/reference/market_to_district.csv`.
4. **Aggregate to district per day.** Modal price is averaged across the district's mandis,
   weighted by arrivals.
5. **Join weather.** Rainfall totals and mean temperatures over the previous 30 and 90 days,
   ending the day before the price date.
6. **Add features.** Lagged prices, harvest-season flag and a time trend.

## Dataset: `data/processed/wheat_district_daily_merged.csv`

| Column | Description |
|---|---|
| `district`, `date` | Row key: one row per district per trading day |
| `modal_price` | **Target.** Arrival-weighted modal wheat price, Rs./quintal |
| `arrivals_tonnes` | Total wheat arrivals in the district that day |
| `log_arrivals` | `log(1 + arrivals_tonnes)` |
| `sharbati_share` | Share of arrivals that is a Sharbati (premium) variety |
| `n_markets` | Number of mandis reporting that day |
| `rain_30d`, `rain_90d` | Rainfall total over the previous 30 / 90 days (mm) |
| `tmax_30d`, `tmax_90d` | Mean daily maximum temperature over the previous 30 / 90 days (°C) |
| `tmin_30d`, `tmin_90d` | Mean daily minimum temperature over the previous 30 / 90 days (°C) |
| `price_lag1` | Modal price on the district's previous trading day |
| `price_prev5_mean` | Mean modal price over the previous five trading days |
| `days_since_prev` | Calendar days since the previous trading day |
| `month`, `harvest_season` | Calendar month; 1 for March–May (rabi wheat arrivals) |
| `days_from_start` | Days since 1 Jan 2022 (time trend) |

`wheat_market_daily_clean.csv` holds the cleaned records before aggregation (one row per
mandi, variety and day, with minimum, maximum and modal price).

## Results

Models are trained on January–October 2022 and tested on November 2022–January 2023
(3,680 district-days). Features are standardised; ridge and lasso penalties are chosen by
time-ordered cross-validation. Errors are in Rs./quintal.

| Feature set | Model | RMSE | MAE | R² |
|---|---|---|---|---|
| Weather + season + supply | Linear | 215 | 168 | -0.15 |
| | Ridge | 307 | 265 | -1.34 |
| | Lasso | 226 | 181 | -0.27 |
| + lagged prices | Linear | 152 | 89 | 0.42 |
| | Ridge | 186 | 140 | 0.14 |
| | Lasso | 170 | 120 | 0.28 |
| Baseline | Previous day's price | 184 | 91 | 0.16 |

**Findings**

- Wheat prices rose steadily through the period (monthly average about 1,920 Rs./quintal in
  January 2022, about 2,610 in January 2023), so the test months lie above the training range.
  Weather, season and supply alone do not carry the models through that shift.
- Adding lagged prices makes linear regression the best model, though only slightly better
  than simply using the previous day's price.
- Ridge and lasso do not improve on ordinary least squares here: the penalty shrinks the
  time-trend coefficient, which matters most when prices keep rising.
- A random train/test split gives a much higher R² (about 0.79 for the weather-only linear
  model) because the model interpolates between neighbouring days. The time-based split is
  the honest measure of forecasting ability.

## Limitations

- **One year of prices.** Seasonality cannot be separated from the 2022 price rise.
- **Market lookup.** `market_to_district.csv` was compiled by hand. Rows flagged `VERIFY` in
  its `check` column have names shared by more than one district and should be confirmed.
  Mandis in the newer Agar Malwa and Niwari districts are assigned to their parent districts
  (Shajapur and Tikamgarh), because the weather data predates the split.
- **Coarse weather.** The source grid is coarse, so neighbouring districts often share identical values and weather
  features are highly correlated.

## Possible next steps

- Add more years of Agmarknet prices.
- Model price changes rather than levels.
- Add minimum support price, national or global wheat prices, and crop production data.
