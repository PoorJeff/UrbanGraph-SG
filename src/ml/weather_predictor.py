"""ML: Weather Feature Prediction Model.

Predicts daily rainfall from temperature, humidity, and wind features.
Demonstrates: feature engineering, model selection, cross-validation, evaluation.

Data: 91 days of NEA weather (2M+ readings aggregated to daily)
Models: LinearRegression baseline → RandomForest → GradientBoosting
Metrics: R², MAE, RMSE
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from src.config import config

logger = logging.getLogger(__name__)


class WeatherPredictor:
    """Train and evaluate ML models for weather prediction."""

    def __init__(self):
        self.data_dir = config.data_dir / "raw" / "nea"
        self.models: dict[str, Any] = {}
        self.results: dict[str, dict[str, float]] = {}
        self.feature_names: list[str] = []

    def prepare_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """Load NEA data and engineer features.

        Returns X (features) and y (target = daily rainfall).
        """
        # Load and aggregate to daily
        daily_dfs = []
        for fname, col in [("rainfall.parquet", "rainfall_mm"),
                            ("temperature.parquet", "temp"),
                            ("humidity.parquet", "humidity"),
                            ("wind_speed.parquet", "wind")]:
            p = self.data_dir / fname
            if not p.exists(): continue
            df = pd.read_parquet(p)
            df["date"] = pd.to_datetime(df["date"])
            if col == "rainfall_mm":
                daily = df.groupby("date")["value"].sum().reset_index()
                daily.columns = ["date", col]
            else:
                daily = df.groupby("date")["value"].mean().reset_index()
                daily.columns = ["date", col]
            daily_dfs.append(daily)

        # Merge all
        merged = daily_dfs[0]
        for d in daily_dfs[1:]:
            merged = merged.merge(d, on="date", how="outer")

        # Feature engineering
        merged = merged.dropna()
        merged["day_of_week"] = merged["date"].dt.dayofweek
        merged["day_of_month"] = merged["date"].dt.day

        # Lag features (rainfall yesterday)
        merged["rain_lag1"] = merged["rainfall_mm"].shift(1)
        merged["temp_lag1"] = merged["temp"].shift(1)
        merged = merged.dropna()

        # Rolling means
        merged["rain_roll3"] = merged["rainfall_mm"].rolling(3).mean()
        merged["temp_roll3"] = merged["temp"].rolling(3).mean()
        merged = merged.dropna()

        self.feature_names = [
            "temp", "humidity", "wind", "day_of_week", "day_of_month",
            "rain_lag1", "temp_lag1", "rain_roll3", "temp_roll3",
        ]
        X = merged[self.feature_names].copy()
        y = merged["rainfall_mm"].copy()

        logger.info(
            "Prepared %d samples, %d features: %s",
            len(X), len(self.feature_names), self.feature_names,
        )
        return X, y

    def train_and_evaluate(self) -> dict[str, dict[str, float]]:
        """Train models and return evaluation metrics."""
        X, y = self.prepare_data()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
        )

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

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

            # Cross-validation
            cv_scores = cross_val_score(model, X_train_s, y_train, cv=5, scoring="r2")

            results[name] = {
                "R2": round(r2, 4),
                "MAE": round(mae, 2),
                "RMSE": round(rmse, 2),
                "CV_R2_mean": round(cv_scores.mean(), 4),
                "CV_R2_std": round(cv_scores.std(), 4),
            }
            self.models[name] = model

            logger.info(
                "  %s: R²=%.4f, MAE=%.2f, RMSE=%.2f, CV_R²=%.4f±%.4f",
                name, r2, mae, rmse, cv_scores.mean(), cv_scores.std(),
            )

        # Feature importance for RandomForest
        if "RandomForest" in self.models:
            importances = self.models["RandomForest"].feature_importances_
            for feat, imp in sorted(zip(self.feature_names, importances), key=lambda x: -x[1]):
                logger.info("  Feature: %s → %.4f", feat, imp)

        self.results = results
        return results

    def get_best_model(self) -> tuple[str, Any]:
        """Return the best performing model."""
        best_name = max(self.results, key=lambda n: self.results[n]["R2"])
        return best_name, self.models[best_name]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    wp = WeatherPredictor()
    results = wp.train_and_evaluate()
    print("\n=== ML Results ===")
    for name, metrics in results.items():
        print(f"{name}: R²={metrics['R2']}, MAE={metrics['MAE']}, RMSE={metrics['RMSE']}, CV_R²={metrics['CV_R2_mean']}±{metrics['CV_R2_std']}")
