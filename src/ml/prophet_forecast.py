"""Prophet Time Series Forecast for Singapore Rainfall.

Predicts next 7 days of rainfall using 91-day NEA data.
Demonstrates: time series forecasting, trend decomposition, uncertainty intervals.
"""

import logging, pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)
REPORTS_DIR = Path(__file__).parent.parent.parent / "reports" / "figures"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False


def forecast(days_ahead=7):
    if not HAS_PROPHET:
        logger.warning("Prophet not installed. Skipping forecast.")
        return None, None

    from src.config import config
    nea = config.data_dir / "raw" / "nea"
    df = pd.read_parquet(nea / "rainfall.parquet")
    df["date"] = pd.to_datetime(df["date"])
    daily = df.groupby("date")["value"].sum().reset_index()
    daily.columns = ["ds", "y"]

    model = Prophet(daily_seasonality=True, weekly_seasonality=True, changepoint_prior_scale=0.05)
    model.fit(daily.tail(60))  # last 60 days for stability

    future = model.make_future_dataframe(periods=days_ahead)
    forecast_df = model.predict(future)

    # Plot
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(daily["ds"], daily["y"], ".", color="#ED2939", alpha=0.4, markersize=6, label="Actual")
    ax.plot(forecast_df["ds"], forecast_df["yhat"], "-", color="#005EC4", linewidth=1.5, label="Forecast")
    ax.fill_between(forecast_df["ds"], forecast_df["yhat_lower"], forecast_df["yhat_upper"],
                    color="#005EC4", alpha=0.1, label="80% CI")
    ax.set_title(f"Singapore Rainfall Forecast — Next {days_ahead} Days (Prophet)", fontsize=13, fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = REPORTS_DIR / "prophet_forecast.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()

    # Summary
    latest = forecast_df.iloc[-days_ahead:][["ds", "yhat", "yhat_lower", "yhat_upper"]]
    trend_text = f"Forecast range: {latest['yhat_lower'].min():.0f}–{latest['yhat_upper'].max():.0f}mm"
    logger.info("Prophet forecast: %s", trend_text)

    return forecast_df, str(path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df, path = forecast(7)
    if df is not None:
        print(f"Forecast saved to {path}")
        print(df[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(7).to_string(index=False))
