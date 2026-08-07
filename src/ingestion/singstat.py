"""SingStat population and demographic data ingestion agent.

Fetches from data.gov.sg CKAN API for planning area population statistics.
If the API is unavailable, falls back to static population data for Singapore's
55 planning areas.

Note: SingStat data via data.gov.sg APIs has been reorganized. Some endpoints
require authentication. Static fallback data is provided for MVP.
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import config
from src.ingestion.base import BaseIngestionAgent

logger = logging.getLogger(__name__)

# Static population data for Singapore planning areas (2024 estimates)
# Source: SingStat Population Trends 2024, Department of Statistics Singapore
# These are approximate figures for MVP purposes
STATIC_POPULATION = [
    {"planning_area": "Ang Mo Kio", "population": 163000, "region": "North-East"},
    {"planning_area": "Bedok", "population": 276000, "region": "East"},
    {"planning_area": "Bishan", "population": 87000, "region": "Central"},
    {"planning_area": "Boon Lay", "population": 1000, "region": "West"},
    {"planning_area": "Bukit Batok", "population": 158000, "region": "West"},
    {"planning_area": "Bukit Merah", "population": 148000, "region": "Central"},
    {"planning_area": "Bukit Panjang", "population": 137000, "region": "West"},
    {"planning_area": "Bukit Timah", "population": 79000, "region": "Central"},
    {"planning_area": "Central Water Catchment", "population": 100, "region": "North"},
    {"planning_area": "Changi", "population": 2000, "region": "East"},
    {"planning_area": "Changi Bay", "population": 0, "region": "East"},
    {"planning_area": "Choa Chu Kang", "population": 186000, "region": "West"},
    {"planning_area": "Clementi", "population": 112000, "region": "West"},
    {"planning_area": "Downtown Core", "population": 4000, "region": "Central"},
    {"planning_area": "Geylang", "population": 110000, "region": "Central"},
    {"planning_area": "Hougang", "population": 227000, "region": "North-East"},
    {"planning_area": "Jurong East", "population": 79000, "region": "West"},
    {"planning_area": "Jurong West", "population": 259000, "region": "West"},
    {"planning_area": "Kallang", "population": 101000, "region": "Central"},
    {"planning_area": "Lim Chu Kang", "population": 100, "region": "North"},
    {"planning_area": "Mandai", "population": 2000, "region": "North"},
    {"planning_area": "Marina East", "population": 0, "region": "Central"},
    {"planning_area": "Marina South", "population": 0, "region": "Central"},
    {"planning_area": "Marine Parade", "population": 45000, "region": "Central"},
    {"planning_area": "Museum", "population": 500, "region": "Central"},
    {"planning_area": "Newton", "population": 8000, "region": "Central"},
    {"planning_area": "North-Eastern Islands", "population": 50, "region": "North-East"},
    {"planning_area": "Novena", "population": 49000, "region": "Central"},
    {"planning_area": "Orchard", "population": 1000, "region": "Central"},
    {"planning_area": "Outram", "population": 20000, "region": "Central"},
    {"planning_area": "Pasir Ris", "population": 152000, "region": "East"},
    {"planning_area": "Paya Lebar", "population": 7000, "region": "Central"},
    {"planning_area": "Pioneer", "population": 100, "region": "West"},
    {"planning_area": "Punggol", "population": 195000, "region": "North-East"},
    {"planning_area": "Queenstown", "population": 98000, "region": "Central"},
    {"planning_area": "River Valley", "population": 10000, "region": "Central"},
    {"planning_area": "Rochor", "population": 13000, "region": "Central"},
    {"planning_area": "Seletar", "population": 3000, "region": "North-East"},
    {"planning_area": "Sembawang", "population": 103000, "region": "North"},
    {"planning_area": "Sengkang", "population": 255000, "region": "North-East"},
    {"planning_area": "Serangoon", "population": 117000, "region": "North-East"},
    {"planning_area": "Simpang", "population": 0, "region": "North"},
    {"planning_area": "Singapore River", "population": 3000, "region": "Central"},
    {"planning_area": "Southern Islands", "population": 2000, "region": "Central"},
    {"planning_area": "Straits View", "population": 0, "region": "Central"},
    {"planning_area": "Sungei Kadut", "population": 100, "region": "North"},
    {"planning_area": "Tampines", "population": 270000, "region": "East"},
    {"planning_area": "Tanglin", "population": 23000, "region": "Central"},
    {"planning_area": "Tengah", "population": 10000, "region": "West"},
    {"planning_area": "Toa Payoh", "population": 106000, "region": "Central"},
    {"planning_area": "Tuas", "population": 100, "region": "West"},
    {"planning_area": "Western Islands", "population": 0, "region": "West"},
    {"planning_area": "Western Water Catchment", "population": 100, "region": "West"},
    {"planning_area": "Woodlands", "population": 255000, "region": "North"},
    {"planning_area": "Yishun", "population": 230000, "region": "North"},
]


class SingStatIngestionAgent(BaseIngestionAgent):
    """Ingest population and demographic statistics.

    Primary source: SingStat via data.gov.sg API.
    Fallback: Static population estimates for 55 planning areas.
    """

    source_name = "singstat"

    def ingest(self) -> dict[str, Path]:
        """Fetch population and household data.

        Returns:
            dict with population_path and optionally household_path
        """
        logger.info("[SINGSTAT] Starting population data ingestion...")

        outputs: dict[str, Path] = {}

        # Attempt API-based population data
        try:
            pop_df = self._fetch_population_api()
            if pop_df is not None and not pop_df.empty:
                outputs["population"] = self.save_dataframe(
                    pop_df, "population.parquet", "population"
                )
                logger.info("[SINGSTAT] API-based population data: %d records", len(pop_df))
        except Exception as e:
            logger.warning("[SINGSTAT] API population fetch failed: %s. Using static fallback.", e)

        # Fallback to static data
        if "population" not in outputs:
            pop_df = pd.DataFrame(STATIC_POPULATION)
            outputs["population"] = self.save_dataframe(
                pop_df, "population.parquet", "population"
            )
            logger.info("[SINGSTAT] Using static population data: %d planning areas", len(pop_df))

        # Attempt household income data
        try:
            household_df = self._fetch_household_api()
            if household_df is not None and not household_df.empty:
                outputs["household"] = self.save_dataframe(
                    household_df, "household.parquet", "household"
                )
        except Exception as e:
            logger.warning("[SINGSTAT] Household data unavailable: %s", e)

        self.write_manifest(
            outputs,
            extra_meta={
                "population_source": "static" if "population" in outputs and "static" in str(outputs.get("population", "")) else "api",
                "record_counts": {
                    name: len(pd.read_parquet(path))
                    for name, path in outputs.items()
                },
            },
        )

        logger.info("[SINGSTAT] Complete: %d datasets", len(outputs))
        return outputs

    def _fetch_population_api(self) -> pd.DataFrame | None:
        """Attempt to fetch population data from data.gov.sg API."""
        # Try the new data.gov.sg developer API
        try:
            import requests
            url = "https://api-open.data.gov.sg/v1/public/api/datasets"
            resp = requests.get(url, timeout=15)
            if resp.status_code == 403:
                logger.info("[SINGSTAT] data.gov.sg v1 API requires authentication. Skipping.")
                return None
            # If accessible, search for population datasets
        except Exception:
            pass

        # Try CKAN API
        try:
            ckan_url = "https://data.gov.sg/api/action/datastore_search"
            # Population-related resource ID (may need updating)
            params = {"resource_id": "d_8b84c4ee58e3cfc0ece0d773c8ca6abc", "limit": 1}
            # This won't give population data, just testing connectivity
            import requests
            resp = requests.get(ckan_url, params=params, timeout=10)
            if resp.status_code == 200:
                logger.info("[SINGSTAT] CKAN API accessible but population dataset ID unknown.")
        except Exception:
            pass

        return None

    def _fetch_household_api(self) -> pd.DataFrame | None:
        """Attempt to fetch household income data from SingStat API."""
        # SingStat household income is typically published as annual reports
        # API access is limited; static summary may be used instead
        return None


if __name__ == "__main__":
    logging.basicConfig(
        level=getattr(logging, config.settings.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    agent = SingStatIngestionAgent()
    agent.ingest()
