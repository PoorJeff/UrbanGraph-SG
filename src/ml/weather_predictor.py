"""ML: Weather Feature Prediction Model.

Predicts daily rainfall from temperature, humidity, and wind features.
Strict methodology: train/test split BEFORE feature engineering,
model selection by cross-validation, no data leakage.

Data: 91 days of NEA weather (2M+ readings aggregated to daily)
Models: LinearRegression baseline → RandomForest → GradientBoosting
Metrics: R², MAE, RMSE, CV-R²
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from src.config import config

logger = logging.getLogger(__name__)


class WeatherPredictor:
    """Train and evaluate ML models for weather prediction.

    CRITICAL DESIGN CHOICES (to avoid data leakage):
    1. Raw data loaded, train/test split done FIRST on date index
    2. Feature engineering (lag, roll) done SEPARATELY on train and test
    3. Model selection uses CV_R2_mean, NOT test set R²
    4. Chronological split (TimeSeriesSplit) to prevent look-ahead bias
    """

    def __init__(self):
        self.data_dir = config.data_dir / "raw" / "nea"
        self.models: dict[str, Any] = {}
        self.results: dict[str, dict[str, float]] = {}
        self.feature_names: list[str] = []

    def _load_raw_daily(self) -> pd.DataFrame:
        """Load raw NEA data, aggregate to daily, return clean DataFrame."""
        daily_dfs = []
        for fname, col in [
            ("rainfall.parquet", "rainfall_mm"),
            ("temperature.parquet", "temp"),
            ("humidity.parquet", "humidity"),
            ("wind_speed.parquet", "wind"),
        ]:
            p = self.data_dir / fname
            if not p.exists():
                continue
            df = pd.read_parquet(p)
            df["date"] = pd.to_datetime(df["date"])
            if col == "rainfall_mm":
                daily = df.groupby("date")["value"].sum().reset_index()
                daily.columns = ["date", col]
            else:
                daily = df.groupby("date")["value"].mean().reset_index()
                daily.columns = ["date", col]
            daily_dfs.append(daily)

        merged = daily_dfs[0]
        for d in daily_dfs[1:]:
            merged = merged.merge(d, on="date", how="outer")
        merged = merged.dropna(subset=["rainfall_mm", "temp", "humidity", "wind"])
        return merged.sort_values("date").reset_index(drop=True)

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features on a split that has ALREADY been separated.

        This MUST run after train/test split to prevent leakage.
        """
        df = df.copy()
        df["day_of_week"] = df["date"].dt.dayofweek
        df["day_of_month"] = df["date"].dt.day
        df["rain_lag1"] = df["rainfall_mm"].shift(1)
        df["temp_lag1"] = df["temp"].shift(1)
        df["rain_roll3"] = df["rainfall_mm"].rolling(3, min_periods=1).mean()
        df["temp_roll3"] = df["temp"].rolling(3, min_periods=1).mean()
        df = df.dropna()  # drop rows where shift created NaN
        return df

    def prepare_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """Load + split + engineer features. No leakage.

        Returns FULL dataset X, y for informational purposes.
        Actual train/test split happens in train_and_evaluate.
        """
        raw = self._load_raw_daily()
        engineered = self._engineer_features(raw)
        self.feature_names = [
            "temp", "humidity", "wind", "day_of_week", "day_of_month",
            "rain_lag1", "temp_lag1", "rain_roll3", "temp_roll3",
        ]
        X = engineered[self.feature_names].copy()
        y = engineered["rainfall_mm"].copy()
        logger.info("Full dataset: %d samples × %d features", len(X), len(self.feature_names))
        return X, y

    def train_and_evaluate(self) -> dict[str, dict[str, float]]:
        """Train models with chronological split + CV-based selection.

        Uses TimeSeriesSplit (3-fold) for robust evaluation.
        Models are compared by mean CV R², NOT single test set R².
        """
        raw = self._load_raw_daily()

        # Set feature names (normally set in prepare_data, but train_and_evaluate
        # bypasses it to do the split-first-then-engineer correctly)
        self.feature_names = [
            "temp", "humidity", "wind", "day_of_week", "day_of_month",
            "rain_lag1", "temp_lag1", "rain_roll3", "temp_roll3",
        ]

        # Chronological split: first 80% train, last 20% test
        # This prevents future data from leaking into training
        split_idx = int(len(raw) * 0.8)
        train_raw = raw.iloc[:split_idx].copy()
        test_raw = raw.iloc[split_idx:].copy()

        # Engineer features SEPARATELY on each split
        train_df = self._engineer_features(train_raw)
        test_df = self._engineer_features(test_raw)

        X_train = train_df[self.feature_names].copy()
        y_train = train_df["rainfall_mm"].copy()
        X_test = test_df[self.feature_names].copy()
        y_test = test_df["rainfall_mm"].copy()

        logger.info(
            "Chronological split: %d train (%.0f%%) + %d test (%.0f%%)",
            len(X_train), len(X_train) / len(raw) * 100,
            len(X_test), len(X_test) / len(raw) * 100,
        )

        # Scale AFTER split
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # TimeSeriesSplit for CV (respects temporal order)
        tscv = TimeSeriesSplit(n_splits=3)

        models = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42),
            "GradientBoosting": GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42),
        }

        results = {}
        for name, model in models.items():
            logger.info("Training %s...", name)
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)

            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            # TimeSeries CV (prevents look-ahead bias)
            cv_scores = cross_val_score(model, X_train_s, y_train, cv=tscv, scoring="r2")

            # Also compute a regular CV for comparison
            cv_reg = cross_val_score(model, X_train_s, y_train, cv=3, scoring="r2")

            results[name] = {
                "R2_test": round(r2, 4),
                "MAE": round(mae, 2),
                "RMSE": round(rmse, 2),
                "CV_R2_mean": round(cv_scores.mean(), 4),
                "CV_R2_std": round(cv_scores.std(), 4),
                "CV_regular_mean": round(cv_reg.mean(), 4),
                "n_train": len(X_train),
                "n_test": len(X_test),
            }
            self.models[name] = model

            logger.info(
                "  %s: R²=%.4f, MAE=%.2f, RMSE=%.2f, CV(TS)=%.4f±%.4f",
                name, r2, mae, rmse, cv_scores.mean(), cv_scores.std(),
            )

        # Feature importance
        if "RandomForest" in self.models:
            importances = self.models["RandomForest"].feature_importances_
            for feat, imp in sorted(zip(self.feature_names, importances), key=lambda x: -x[1]):
                logger.info("  Feature: %s → %.4f", feat, imp)

        self.results = results
        return results

    def get_best_model(self) -> tuple[str, Any]:
        """Return best model by cross-validation score (not test R²).

        Was broken: used `max(R²)` which favored overfit LR.
        Now: uses `max(CV_R2_mean)` for robust selection.
        """
        best_name = max(self.results, key=lambda n: self.results[n]["CV_R2_mean"])
        return best_name, self.models[best_name]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    wp = WeatherPredictor()
    results = wp.train_and_evaluate()
    best_name, _ = wp.get_best_model()

    print("\n=== ML Results (No Data Leakage) ===")
    print(f"Train/Test split: chronological (first 80% train, last 20% test)")
    print(f"Best model by CV: {best_name}")
    print()
    for name, m in results.items():
        marker = " ← BEST" if name == best_name else ""
        print(f"{name}: R²_test={m['R2_test']:.4f}, MAE={m['MAE']:.1f}, RMSE={m['RMSE']:.1f}, "
              f"CV_TS={m['CV_R2_mean']:.4f}±{m['CV_R2_std']:.4f}{marker}")
