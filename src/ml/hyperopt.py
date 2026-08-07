"""Optuna Hyperparameter Optimization for WeatherPredictor.

Searches for best RandomForest params automatically.
"""
import logging
logger = logging.getLogger(__name__)


def run_optimization(n_trials=30):
    """Run Optuna hyperparameter search. Returns best params dict."""
    try:
        import optuna
    except ImportError:
        logger.warning("optuna not installed. pip install optuna")
        return None

    from src.ml.weather_predictor import WeatherPredictor
    wp = WeatherPredictor()
    raw = wp._load_raw_daily()
    split_idx = int(len(raw) * 0.8)
    train_df = wp._engineer_features(raw.iloc[:split_idx].copy())
    X = train_df[wp.feature_names].values
    y = train_df["rainfall_mm"].values

    def objective(trial):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import cross_val_score
        model = RandomForestRegressor(
            n_estimators=trial.suggest_int("n_estimators", 50, 200),
            max_depth=trial.suggest_int("max_depth", 3, 15),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
            min_samples_leaf=trial.suggest_float("min_samples_leaf", 0.01, 0.1),
            random_state=42,
        )
        scores = cross_val_score(model, X, y, cv=3, scoring="r2")
        return float(scores.mean())

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    logger.info("Optuna best CV R²=%.4f, params=%s", study.best_value, study.best_params)
    return {"best_cv_r2": round(study.best_value, 4), "best_params": study.best_params}
