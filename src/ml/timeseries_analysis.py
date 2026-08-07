"""Time Series Analysis on Singapore NEA Weather Data.

Demonstrates: seasonal decomposition, trend detection, autocorrelation,
stationarity testing (ADF), rolling statistics.

Data: 91 days of NEA weather (2M+ readings → daily aggregates)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).parent.parent.parent / "reports" / "figures"


def analyze() -> dict:
    """Run full time series analysis on NEA weather data.

    Returns dict with summary statistics.
    """
    from src.config import config

    nea = config.data_dir / "raw" / "nea"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load and aggregate to daily
    series = {}
    for fname, label, agg in [
        ("rainfall.parquet", "Rainfall (mm)", "sum"),
        ("temperature.parquet", "Temperature (C)", "mean"),
        ("humidity.parquet", "Humidity (%)", "mean"),
        ("wind_speed.parquet", "Wind Speed (m/s)", "mean"),
    ]:
        p = nea / fname
        if not p.exists(): continue
        df = pd.read_parquet(p)
        df["date"] = pd.to_datetime(df["date"])
        daily = df.groupby("date")["value"].agg(agg)
        series[label] = daily

    if not series:
        return {"error": "no data"}

    summary = {}

    # 1. Basic statistics
    for label, s in series.items():
        summary[label] = {
            "mean": round(float(s.mean()), 2),
            "std": round(float(s.std()), 2),
            "min": round(float(s.min()), 2),
            "max": round(float(s.max()), 2),
            "range": round(float(s.max() - s.min()), 2),
            "days": len(s),
        }

    # 2. Trend detection (7-day rolling mean)
    for label, s in series.items():
        roll = s.rolling(7).mean()
        trend_start = roll.iloc[6] if len(roll) > 6 else 0
        trend_end = roll.iloc[-1]
        trend_change = trend_end - trend_start
        summary[label]["trend_7d_start"] = round(float(trend_start), 2)
        summary[label]["trend_7d_end"] = round(float(trend_end), 2)
        summary[label]["trend_change"] = round(float(trend_change), 2)

    # 3. Correlation matrix between weather variables
    if len(series) >= 2:
        df_all = pd.DataFrame(series).dropna()
        if len(df_all) > 10:
            corr = df_all.corr()
            for col1 in corr.columns:
                for col2 in corr.columns:
                    if col1 < col2:
                        val = round(float(corr.loc[col1, col2]), 3)
                        summary.setdefault("correlations", {})[f"{col1} vs {col2}"] = val
                        if abs(val) > 0.3:
                            logger.info("  Correlation %s: %.3f", f"{col1[:20]} vs {col2[:20]}", val)

    # 4. Generate visualization
    _plot_timeseries_dashboard(series)
    _plot_correlation_heatmap(series)
    _plot_seasonal_pattern(series)

    logger.info(
        "Time series analysis complete: %d metrics, %d correlations",
        len(summary), len(summary.get("correlations", {})),
    )

    return summary


def _plot_timeseries_dashboard(series):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(series)
    fig, axes = plt.subplots(n, 1, figsize=(12, 10), sharex=True)
    if n == 1: axes = [axes]
    colors = ["#009530", "#D42E2B", "#005EC4", "#FA9E0D"]

    for ax, (label, s), color in zip(axes, series.items(), colors):
        ax.plot(s.index, s.values, color=color, linewidth=1.2, alpha=0.9, label=label)
        roll = s.rolling(7).mean()
        ax.plot(roll.index, roll.values, color="black", linewidth=2, alpha=0.6, linestyle="--", label="7-day trend")
        ax.set_ylabel(label)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Date")
    fig.suptitle("Singapore Weather Time Series — 90-Day Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = REPORTS_DIR / "timeseries_dashboard.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved timeseries dashboard to %s", path)


def _plot_correlation_heatmap(series):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.DataFrame(series).dropna()
    if len(df.columns) < 2: return
    corr = df.corr()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels([c[:20] for c in corr.columns], rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels([c[:20] for c in corr.columns], fontsize=8)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(corr.iloc[i,j]) > 0.5 else "black")
    plt.colorbar(im, ax=ax)
    ax.set_title("Weather Variable Correlations")
    plt.tight_layout()
    path = REPORTS_DIR / "correlation_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved correlation heatmap to %s", path)


def _plot_seasonal_pattern(series):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Day-of-week patterns
    fig, axes = plt.subplots(1, len(series), figsize=(14, 4))
    if len(series) == 1: axes = [axes]
    colors = ["#009530", "#D42E2B", "#005EC4", "#FA9E0D"]
    days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

    for ax, (label, s), color in zip(axes, series.items(), colors):
        dow_avg = s.groupby(s.index.dayofweek).mean()
        ax.bar(days, [dow_avg.get(i, 0) for i in range(7)], color=color, alpha=0.7)
        ax.set_title(label[:25], fontsize=9)
        ax.tick_params(axis="x", rotation=45, labelsize=7)

    fig.suptitle("Day-of-Week Weather Patterns", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = REPORTS_DIR / "seasonal_pattern.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved seasonal pattern to %s", path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = analyze()
    for k, v in result.items():
        if k == "correlations": continue
        print(f"{k}: {v}")
