"""NEA Weather ingestion agent.

Fetches weather data from data.gov.sg v1 environment APIs.
Endpoints: rainfall, air-temperature, psi, wind-speed, humidity, uv-index.

Historical data requires date queries; we fetch one date at a time
and aggregate into parquet files.
"""

import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from src.config import config
from src.ingestion.base import BaseIngestionAgent

logger = logging.getLogger(__name__)

# API base URL
DATA_GOV_SG_V1 = "https://api.data.gov.sg/v1"

# NEA environment endpoints
NEA_ENDPOINTS = {
    "rainfall": "/environment/rainfall",
    "temperature": "/environment/air-temperature",
    "humidity": "/environment/relative-humidity",
    "wind_speed": "/environment/wind-speed",
    "wind_direction": "/environment/wind-direction",
    "psi": "/environment/psi",
    "uv_index": "/environment/uv-index",
}


class NEAWeatherIngestionAgent(BaseIngestionAgent):
    """Ingest NEA weather data from data.gov.sg."""

    source_name = "nea"

    def __init__(self):
        super().__init__()
        self.start_date = date(self.settings.data_start_year, 1, 1)
        self.end_date = date(self.settings.data_end_year, 12, 31)

    def ingest(self) -> dict[str, Path]:
        """Fetch all weather data for the configured date range.

        Returns:
            dict mapping dataset name to output file path
        """
        logger.info(
            "[NEA] Starting weather data ingestion: %s to %s",
            self.start_date, self.end_date,
        )

        outputs: dict[str, Path] = {}
        record_counts: dict[str, int] = {}

        for name, endpoint in NEA_ENDPOINTS.items():
            logger.info("[NEA] Fetching %s data...", name)
            try:
                records = self._fetch_historical(endpoint)
                df = pd.DataFrame(records)

                if df.empty:
                    logger.warning("[NEA] No %s records found", name)
                    continue

                output_path = self.save_dataframe(
                    df, f"{name}.parquet", name
                )
                outputs[name] = output_path
                record_counts[name] = len(df)

            except Exception as e:
                logger.error("[NEA] Failed to fetch %s: %s", name, e)
                # Don't block other endpoints
                continue

        self.write_manifest(
            outputs,
            extra_meta={
                "date_range": f"{self.start_date} to {self.end_date}",
                "record_counts": record_counts,
            },
        )

        total = sum(record_counts.values())
        logger.info(
            "[NEA] Complete: %d total records across %d datasets",
            total, len(outputs),
        )
        return outputs

    def _fetch_historical(self, endpoint: str) -> list[dict[str, Any]]:
        """Fetch historical data by iterating through dates.

        data.gov.sg environment APIs return data for a specific date.
        We iterate through each date in the configured range.

        Returns:
            List of reading dictionaries with date + station data
        """
        all_readings: list[dict[str, Any]] = []
        current = self.start_date

        today = datetime.now(timezone.utc).date()

        while current <= min(self.end_date, today):
            date_str = current.isoformat()
            params = {"date": date_str}

            try:
                data = self.fetch_json(
                    f"{DATA_GOV_SG_V1}{endpoint}",
                    params=params,
                    timeout=30,
                )
            except RuntimeError:
                # One day's failure shouldn't stop the whole pipeline
                logger.debug("[NEA] No data for %s on %s", endpoint, date_str)
                current += timedelta(days=1)
                continue

            items = data.get("items", [])
            metadata = data.get("metadata", {})

            for item in items:
                timestamp = item.get("timestamp", "")
                readings = item.get("readings", [])

                for reading in readings:
                    record = {
                        "date": date_str,
                        "timestamp": timestamp,
                        "station_id": reading.get("station_id", ""),
                        "value": reading.get("value"),
                    }
                    all_readings.append(record)

            current += timedelta(days=1)

            # Progress logging every 30 days
            if len(all_readings) % 1000 == 0 and len(all_readings) > 0:
                logger.debug(
                    "[NEA] %s: %d readings so far (at %s)...",
                    endpoint, len(all_readings), date_str,
                )

        logger.info(
            "[NEA] %s: fetched %d readings across %d days",
            endpoint, len(all_readings),
            (min(self.end_date, today) - self.start_date).days + 1,
        )
        return all_readings

    def _generate_weather_events(
        self, raw_data: dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Aggregate daily weather data into WeatherEvent entities.

        Computes: rainfall_total, temperature_max/min, humidity_avg, psi_max
        per day per station (or Singapore-wide).
        """
        # Combine rainfall data
        if "rainfall" in raw_data and not raw_data["rainfall"].empty:
            rainfall = raw_data["rainfall"].copy()
            daily_rain = (
                rainfall.groupby(["date", "station_id"])["value"]
                .sum()
                .reset_index()
                .rename(columns={"value": "rainfall_mm"})
            )
        else:
            daily_rain = pd.DataFrame(
                columns=["date", "station_id", "rainfall_mm"]
            )

        # Combine temperature data
        if "temperature" in raw_data and not raw_data["temperature"].empty:
            temp = raw_data["temperature"].copy()
            daily_temp = (
                temp.groupby(["date", "station_id"])["value"]
                .agg(["max", "min", "mean"])
                .reset_index()
            )
            daily_temp.columns = [
                "date", "station_id",
                "temperature_max", "temperature_min", "temperature_mean",
            ]
        else:
            daily_temp = pd.DataFrame(
                columns=[
                    "date", "station_id",
                    "temperature_max", "temperature_min", "temperature_mean",
                ]
            )

        return daily_rain  # placeholder; full implementation in processing stage


if __name__ == "__main__":
    logging.basicConfig(
        level=getattr(logging, config.settings.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    agent = NEAWeatherIngestionAgent()
    # For testing, fetch just 7 days
    agent.start_date = date.today() - timedelta(days=7)
    agent.end_date = date.today()
    agent.ingest()
