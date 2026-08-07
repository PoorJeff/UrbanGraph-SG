"""ML Visualization — automated plots for model evaluation and insights.

Generates:
- Feature importance bar chart
- Prediction vs Actual scatter plot
- Learning curve
- Weather time-series dashboard
Saves to reports/figures/ directory.
"""

import logging
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).parent.parent.parent / "reports" / "figures"

def plot_feature_importance(feature_names, importances, title="Feature Importance"):
    """Generate and save a feature importance bar chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    sorted_idx = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([feature_names[i] for i in sorted_idx], [importances[i] for i in sorted_idx],
            color="#ED2939")
    ax.set_xlabel("Importance")
    ax.set_title(title)
    plt.tight_layout()
    path = REPORTS_DIR / "feature_importance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved feature importance to %s", path)
    return str(path)

def plot_predictions(y_true, y_pred, title="Prediction vs Actual"):
    """Generate prediction vs actual scatter plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.6, color="#ED2939", s=30)
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], "k--", alpha=0.3)
    ax.set_xlabel("Actual Rainfall (mm)")
    ax.set_ylabel("Predicted Rainfall (mm)")
    ax.set_title(title)
    plt.tight_layout()
    path = REPORTS_DIR / "prediction_vs_actual.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved prediction plot to %s", path)
    return str(path)

def plot_weather_dashboard():
    """Generate a weather time-series dashboard from NEA data."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    from src.config import config
    nea = config.data_dir / "raw" / "nea"

    dfs = {}
    for fname, label in [("rainfall.parquet","Rainfall"),("temperature.parquet","Temperature"),
                          ("humidity.parquet","Humidity"),("wind_speed.parquet","Wind")]:
        p = nea / fname
        if not p.exists(): continue
        df = pd.read_parquet(p)
        df["date"] = pd.to_datetime(df["date"])
        if fname == "rainfall.parquet":
            daily = df.groupby("date")["value"].sum()
        else:
            daily = df.groupby("date")["value"].mean()
        dfs[label] = daily

    fig, axes = plt.subplots(len(dfs), 1, figsize=(12, 10), sharex=True)
    colors = ["#009530", "#D42E2B", "#005EC4", "#FA9E0D"]
    for ax, (label, series), color in zip(axes, dfs.items(), colors):
        ax.plot(series.index, series.values, color=color, linewidth=0.8, alpha=0.8)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Date")
    fig.suptitle("Singapore Weather Dashboard — 90 Days", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = REPORTS_DIR / "weather_dashboard.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved weather dashboard to %s", path)
    return str(path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    plot_weather_dashboard()
    print("Plots saved to:", REPORTS_DIR)
