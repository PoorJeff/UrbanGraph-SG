"""LTA DataMall ingestion agent.

Fetches static and real-time transport data from LTA DataMall HTTPS API.
Endpoints: BusStops, BusServices, BusRoutes, BusArrival, TaxiAvailability.

Note: MRT station data is NOT available via LTA DataMall (endpoint retired).
MRT stations are sourced from OneMap API instead.
"""

import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.config import config
from src.ingestion.base import BaseIngestionAgent

logger = logging.getLogger(__name__)

# LTA DataMall API base (HTTPS — HTTP is retired)
LTA_API_BASE = "https://datamall2.mytransport.sg/ltaodataservice"

# Available static datasets
STATIC_ENDPOINTS = {
    "bus_stops": "/BusStops",
    "bus_services": "/BusServices",
    "bus_routes": "/BusRoutes",
}


class LTAIngestionAgent(BaseIngestionAgent):
    """Ingest LTA DataMall transport data."""

    source_name = "lta"

    def __init__(self):
        super().__init__()
        self.account_key = self.settings.lta_account_key
        if not self.account_key:
            raise ValueError(
                "LTA_ACCOUNT_KEY not set. "
                "Set it in .env or environment variables."
            )

        self.headers = {
            "AccountKey": self.account_key,
            "Accept": "application/json",
        }

    def ingest(self) -> dict[str, Path]:
        """Fetch all LTA static datasets.

        Returns:
            dict mapping dataset name to output file path
        """
        logger.info("[LTA] Starting data ingestion...")

        outputs: dict[str, Path] = {}
        record_counts: dict[str, int] = {}

        for name, endpoint in STATIC_ENDPOINTS.items():
            logger.info("[LTA] Fetching %s...", name)
            try:
                url = f"{LTA_API_BASE}{endpoint}"
                records = self.fetch_paginated_odata(url, headers=self.headers)
                df = pd.DataFrame(records)

                if df.empty:
                    logger.warning("[LTA] No %s records found", name)
                    continue

                output_path = self.save_dataframe(
                    df, f"{name}.parquet", name
                )
                outputs[name] = output_path
                record_counts[name] = len(df)

                logger.info(
                    "[LTA] %s: %d records saved",
                    name, len(df),
                )

            except Exception as e:
                logger.error("[LTA] Failed to fetch %s: %s", name, e)
                continue

        self.write_manifest(
            outputs,
            extra_meta={
                "api_base": LTA_API_BASE,
                "record_counts": record_counts,
            },
        )

        total = sum(record_counts.values())
        logger.info(
            "[LTA] Complete: %d total records across %d datasets",
            total, len(outputs),
        )
        return outputs

    def fetch_bus_arrival(
        self, bus_stop_code: str
    ) -> Optional[dict[str, Any]]:
        """Fetch real-time bus arrival data for a specific bus stop.

        Note: This is real-time data; use sparingly.
        For MVP, we only ingest static data.
        """
        url = f"{LTA_API_BASE}/BusArrivalv2"
        params = {"BusStopCode": bus_stop_code}
        try:
            return self.fetch_json(url, headers=self.headers, params=params)
        except RuntimeError:
            logger.debug("[LTA] Bus arrival fetch failed for %s", bus_stop_code)
            return None

    def fetch_taxi_availability(self) -> Optional[dict[str, Any]]:
        """Fetch real-time taxi availability data.

        Note: Snapshot data; for MVP we prioritize static datasets.
        """
        url = f"{LTA_API_BASE}/TaxiAvail"
        try:
            return self.fetch_json(url, headers=self.headers)
        except RuntimeError:
            logger.debug("[LTA] Taxi availability fetch failed")
            return None


if __name__ == "__main__":
    logging.basicConfig(
        level=getattr(logging, config.settings.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    agent = LTAIngestionAgent()
    agent.ingest()
