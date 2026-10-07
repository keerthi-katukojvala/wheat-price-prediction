"""
Step 2: fit linear, ridge and lasso regression on the merged dataset.

Two feature sets are compared on a time-based split (train up to TRAIN_END,
test afterwards):

  A. weather + season + supply
  B. the same plus lagged prices

Writes results/model_results.csv and results/model_coefficients.csv.

Run from anywhere:  python src/train_models.py   (after build_dataset.py)
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config as cfg

warnings.filterwarnings("ignore", category=ConvergenceWarning)

TARGET = "modal_price"
BASE_FEATURES = ["log_arrivals", "sharbati_share", "n_markets", "harvest_season",
                 "days_from_start", "rain_30d", "rain_90d", "tmax_30d", "tmax_90d",
                 "tmin_30d", "tmin_90d"]
LAG_FEATURES = ["price_lag1", "price_prev5_mean", "days_since_prev"]
ALPHAS = np.logspace(-3, 3, 40)


def score(name, feature_set, y_true, y_pred, n_train, alpha=np.nan):
    return {"feature_set": feature_set, "model": name,
            "train_rows": n_train, "test_rows": len(y_true),
            "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
            "alpha": alpha}


def evaluate(df, features, feature_set):
    df = df.dropna(subset=features + [TARGET]).sort_values("date")
    X = pd.get_dummies(df[features + ["district"]], columns=["district"],
                       drop_first=True, dtype=float)
    y = df[TARGET]
    is_train = df["date"] <= cfg.TRAIN_END
    X_train, X_test, y_train, y_test = X[is_train], X[~is_train], y[is_train], y[~is_train]

    cv = TimeSeriesSplit(n_splits=5)                     # keeps validation folds in the future
    models = {
        "Linear": LinearRegression(),
        "Ridge": RidgeCV(alphas=ALPHAS, cv=cv),
        "Lasso": LassoCV(alphas=ALPHAS, cv=cv, max_iter=50_000),
    }

    scores, coefficients = [], {}
    for name, model in models.items():
        pipeline = make_pipeline(StandardScaler(), model).fit(X_train, y_train)
        scores.append(score(name, feature_set, y_test, pipeline.predict(X_test),
                            len(X_train), getattr(model, "alpha_", np.nan)))
        coefficients[name] = pd.Series(model.coef_, index=X.columns)

    # Naive baselines for comparison.
    scores.append(score("Baseline: training mean", feature_set, y_test,
                        np.full(len(y_test), y_train.mean()), len(X_train)))
    if "price_lag1" in features:
        scores.append(score("Baseline: previous day's price", feature_set, y_test,
                            df.loc[~is_train, "price_lag1"], len(X_train)))

    coef_table = pd.DataFrame(coefficients)
    coef_table = coef_table[~coef_table.index.str.startswith("district_")]
    coef_table.insert(0, "feature_set", feature_set)
    return pd.DataFrame(scores), coef_table.reset_index(names="feature")


def main():
    if not cfg.DISTRICT_DAILY_FILE.exists():
        raise FileNotFoundError("Run src/build_dataset.py first.")
    df = pd.read_csv(cfg.DISTRICT_DAILY_FILE, parse_dates=["date"])

    runs = [evaluate(df, BASE_FEATURES, "Weather + season + supply"),
            evaluate(df, BASE_FEATURES + LAG_FEATURES, "Same + lagged prices")]
    results = pd.concat([r[0] for r in runs], ignore_index=True)
    coefficients = pd.concat([r[1] for r in runs], ignore_index=True)

    cfg.RESULTS_DIR.mkdir(exist_ok=True)
    results.round(4).to_csv(cfg.RESULTS_DIR / "model_results.csv", index=False)
    coefficients.round(4).to_csv(cfg.RESULTS_DIR / "model_coefficients.csv", index=False)

    print(f"Test period: after {cfg.TRAIN_END}  (errors in Rs./quintal)\n")
    print(results.round(3).to_string(index=False))
    print("\nStandardised coefficients (district dummies omitted)\n")
    print(coefficients.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
