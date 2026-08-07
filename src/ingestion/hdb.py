"""HDB resale and rental data ingestion agent.

Fetches from data.gov.sg CKAN API (not the v1 REST API which is deprecated).
Resource IDs for HDB datasets are hardcoded; they may need updating over time.
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import config
from src.ingestion.base import BaseIngestionAgent

logger = logging.getLogger(__name__)

# data.gov.sg CKAN API base
CKAN_API_BASE = "https://data.gov.sg/api/action/datastore_search"

# Resource IDs for HDB datasets (from data.gov.sg)
# These are stable but may change if the dataset is republished
HDB_RESOURCES = {
    "resale_prices": "d_8b84c4ee58e3cfc0ece0d773c8ca6abc",
}

# Maximum records per API call (CKAN limit)
CKAN_LIMIT = 100


class HDBIngestionAgent(BaseIngestionAgent):
    """Ingest HDB resale price and rental data from data.gov.sg."""

    source_name = "hdb"

    def ingest(self) -> dict[str, Path]:
        """Fetch HDB resale prices for the configured date range.

        Returns:
            dict mapping dataset name to output file path
        """
        logger.info("[HDB] Starting HDB data ingestion...")

        outputs: dict[str, Path] = {}
        record_counts: dict[str, int] = {}

        for name, resource_id in HDB_RESOURCES.items():
            logger.info("[HDB] Fetching %s (resource: %s)...", name, resource_id)
            try:
                records = self._fetch_ckan_dataset(resource_id)
                df = pd.DataFrame(records)

                if df.empty:
                    logger.warning("[HDB] No %s records found", name)
                    continue

                output_path = self.save_dataframe(
                    df, f"{name}.parquet", name
                )
                outputs[name] = output_path
                record_counts[name] = len(df)

                logger.info(
                    "[HDB] %s: %d records saved",
                    name, len(df),
                )

            except Exception as e:
                logger.error("[HDB] Failed to fetch %s: %s", name, e)
                continue

        self.write_manifest(
            outputs,
            extra_meta={
                "api_base": CKAN_API_BASE,
                "record_counts": record_counts,
            },
        )

        total = sum(record_counts.values())
        logger.info(
            "[HDB] Complete: %d total records across %d datasets",
            total, len(outputs),
        )
        return outputs

    def _fetch_ckan_dataset(
        self,
        resource_id: str,
        max_records: int = 100_000,
    ) -> list[dict[str, Any]]:
        """Fetch all records from a CKAN datastore resource with pagination.

        Args:
            resource_id: CKAN resource ID
            max_records: Safety cap on total records

        Returns:
            List of record dicts with CKAN _id field removed
        """
        all_records: list[dict[str, Any]] = []
        offset = 0

        while len(all_records) < max_records:
            params = {
                "resource_id": resource_id,
                "limit": CKAN_LIMIT,
                "offset": offset,
            }

            data = self.fetch_json(CKAN_API_BASE, params=params)
            result = data.get("result", {})
            records = result.get("records", [])

            if not records:
                break

            # Remove CKAN internal fields
            clean_records = [
                {k: v for k, v in r.items() if k != "_id"}
                for r in records
            ]
            all_records.extend(clean_records)

            total = result.get("total", 0)
            if offset + CKAN_LIMIT >= total:
                break

            offset += CKAN_LIMIT

            if len(all_records) % 5000 == 0:
                logger.debug(
                    "[HDB] Fetched %d/%d records...",
                    len(all_records), total,
                )

        logger.info(
            "[HDB] Fetched %d records from resource %s",
            len(all_records), resource_id,
        )
        return all_records


if __name__ == "__main__":
    logging.basicConfig(
        level=getattr(logging, config.settings.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    agent = HDBIngestionAgent()
    agent.ingest()
