"""MLOps pipeline — experiment tracking and model evaluation.

Uses MLflow for experiment tracking when available.
Otherwise provides a lightweight evaluation framework.

Demonstrates: experiment tracking, model registry, evaluation metrics,
reproducible ML workflow.
"""

import json, logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
MLOPS_DIR = Path(__file__).parent.parent.parent / "reports" / "mlops"
MLOPS_DIR.mkdir(parents=True, exist_ok=True)

# Try MLflow
try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False


class ExperimentTracker:
    """Lightweight experiment tracker (MLflow-compatible interface).

    Tracks: model runs, hyperparameters, metrics, artifacts.
    Saves to reports/mlops/ for reproducibility.
    """

    def __init__(self, experiment_name: str = "urbangraph-sg"):
        self.experiment_name = experiment_name
        self.runs: list[dict] = []
        self._current_run: dict | None = None

        if HAS_MLFLOW:
            try:
                mlflow.set_tracking_uri(f"file://{MLOPS_DIR}/mlruns")
                mlflow.set_experiment(experiment_name)
                logger.info("MLflow tracking: %s", MLOPS_DIR / "mlruns")
            except Exception:
                pass

    def start_run(self, run_name: str, params: dict[str, Any] = None):
        self._current_run = {
            "run_name": run_name,
            "params": params or {},
            "metrics": {},
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "running",
        }
        if HAS_MLFLOW:
            try:
                mlflow.start_run(run_name=run_name)
                if params:
                    mlflow.log_params(params)
            except Exception:
                pass

    def log_metric(self, key: str, value: float, step: int = 0):
        if self._current_run:
            self._current_run["metrics"][key] = value
        if HAS_MLFLOW:
            try:
                mlflow.log_metric(key, value, step=step)
            except Exception:
                pass

    def log_metrics(self, metrics: dict[str, float]):
        for k, v in metrics.items():
            self.log_metric(k, v)

    def log_artifact(self, path: str):
        if HAS_MLFLOW:
            try:
                mlflow.log_artifact(path)
            except Exception:
                pass

    def end_run(self):
        if self._current_run:
            self._current_run["ended_at"] = datetime.now(timezone.utc).isoformat()
            self._current_run["status"] = "completed"
            self.runs.append(self._current_run)
            self._save()
            self._current_run = None
        if HAS_MLFLOW:
            try:
                mlflow.end_run()
            except Exception:
                pass

    def _save(self):
        path = MLOPS_DIR / f"experiments_{datetime.now().strftime('%Y%m%d')}.json"
        # Load existing, append new run
        existing = []
        if path.exists():
            try:
                existing = json.loads(path.read_text())
            except Exception:
                pass
        existing.extend(self.runs)
        path.write_text(json.dumps(existing, indent=2, default=str))


def run_ml_evaluation():
    """Run a complete ML evaluation pipeline and track results."""
    tracker = ExperimentTracker("urbangraph-weather-prediction")

    # Run 1: Baseline LinearRegression
    tracker.start_run("LinearRegression_baseline", {"model": "LinearRegression", "features": 9})
    from src.ml.weather_predictor import WeatherPredictor
    wp = WeatherPredictor()
    X, y = wp.prepare_data()
    results = wp.train_and_evaluate()
    lr = results.get("LinearRegression", {})
    tracker.log_metrics({f"LR_{k}": v for k, v in lr.items() if isinstance(v, (int, float))})
    tracker.end_run()

    # Run 2: RandomForest
    tracker.start_run("RandomForest_tuned", {"model": "RandomForest", "n_estimators": 100, "max_depth": 8})
    rf = results.get("RandomForest", {})
    tracker.log_metrics({f"RF_{k}": v for k, v in rf.items() if isinstance(v, (int, float))})
    tracker.end_run()

    # Run 3: GradientBoosting
    tracker.start_run("GradientBoosting", {"model": "GradientBoosting", "n_estimators": 100})
    gb = results.get("GradientBoosting", {})
    tracker.log_metrics({f"GB_{k}": v for k, v in gb.items() if isinstance(v, (int, float))})
    tracker.end_run()

    # Generate comparison report
    report = generate_model_card(results)
    logger.info("ML evaluation complete: %d experiments tracked", len(tracker.runs))
    return report


def generate_model_card(results: dict[str, dict]) -> dict:
    """Generate a model card for the project."""
    best = max(results, key=lambda n: results[n].get("CV_R2_mean", results[n].get("R2_test", 0)))
    card = {
        "project": "UrbanGraph-SG Weather Prediction",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "NEA Singapore weather, 91 days, 4 variables, 78 stations (daily aggregation)",
        "task": "Regression — predict total daily rainfall (sum across stations)",
        "best_model": best,
        "split": "Chronological: first 80% train, last 20% test (no data leakage)",
        "models": {},
    }
    for name, metrics in results.items():
        card["models"][name] = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}

    path = MLOPS_DIR / "model_card.json"
    path.write_text(json.dumps(card, indent=2))
    logger.info("Model card saved to %s", path)
    return card


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    report = run_ml_evaluation()
    for model, metrics in report["models"].items():
        print(f"{model}: {metrics}")
