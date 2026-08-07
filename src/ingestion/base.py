"""Base ingestion agent with retry logic, manifest writing, and logging."""

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import config

logger = logging.getLogger(__name__)


class BaseIngestionAgent(ABC):
    """Base class for all data ingestion agents.

    Provides:
    - HTTP session with retry logic (exponential backoff, max 3 attempts)
    - Manifest file generation (timestamp, record count, file hash)
    - Parquet output with consistent schema
    - OData pagination support
    """

    source_name: str = "base"

    def __init__(self):
        self.settings = config.settings
        self.raw_dir = config.data_dir / "raw" / self.source_name
        self.manifest_dir = config.data_dir / "manifests"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_dir.mkdir(parents=True, exist_ok=True)

        # Create session with retry
        self.session = self._build_session()

    def _build_session(self) -> requests.Session:
        """Build a requests session with exponential backoff retry."""
        session = requests.Session()

        retry_strategy = Retry(
            total=3,
            backoff_factor=5,  # 1s, 5s, 25s
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        return session

    @abstractmethod
    def ingest(self) -> dict[str, Path]:
        """Run the ingestion. Returns dict of dataset_name -> output_path."""
        ...

    def fetch_json(
        self,
        url: str,
        headers: Optional[dict[str, str]] = None,
        params: Optional[dict[str, Any]] = None,
        timeout: int = 60,
    ) -> dict[str, Any]:
        """Fetch JSON from an API endpoint with error handling.

        Raises RuntimeError on repeated failures.
        """
        for attempt in range(1, 4):
            try:
                resp = self.session.get(
                    url, headers=headers, params=params, timeout=timeout
                )
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.RequestException as e:
                logger.warning(
                    "[%s] Attempt %d/3 failed for %s: %s",
                    self.source_name.upper(), attempt, url, e,
                )
                if attempt == 3:
                    raise RuntimeError(
                        f"[{self.source_name.upper()}] Failed to fetch {url} after 3 attempts"
                    ) from e
                time.sleep(5 ** (attempt - 1))  # 1s, 5s, 25s

        raise RuntimeError("Unreachable")

    def fetch_paginated_odata(
        self,
        base_url: str,
        headers: Optional[dict[str, str]] = None,
        page_size: int = 500,
        max_records: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Fetch all records from an OData API with pagination.

        Args:
            base_url: OData endpoint URL
            headers: HTTP headers
            page_size: Number of records per page (max 500 for LTA)
            max_records: Maximum total records to fetch (None = all)

        Returns:
            List of all records across all pages
        """
        all_records: list[dict[str, Any]] = []
        skip = 0

        while True:
            params = {"$top": page_size, "$skip": skip}
            data = self.fetch_json(base_url, headers=headers, params=params)
            records = data.get("value", [])

            if not records:
                break

            all_records.extend(records)

            if max_records and len(all_records) >= max_records:
                all_records = all_records[:max_records]
                break

            if len(records) < page_size:
                # Last page
                break

            skip += page_size
            logger.debug(
                "[%s] Paginated: fetched %d records so far...",
                self.source_name.upper(), len(all_records),
            )

        logger.info(
            "[%s] Pagination complete: %d total records from %s",
            self.source_name.upper(), len(all_records), base_url,
        )
        return all_records

    def save_dataframe(
        self,
        df: pd.DataFrame,
        filename: str,
        dataset_name: str,
        record_count: Optional[int] = None,
    ) -> Path:
        """Save a DataFrame as Parquet and return the output path."""
        output_path = self.raw_dir / filename
        df.to_parquet(output_path, index=False)

        actual_count = record_count if record_count is not None else len(df)
        logger.info(
            "[%s] Saved %s: %d records -> %s",
            self.source_name.upper(), dataset_name, actual_count, output_path,
        )
        return output_path

    def write_manifest(
        self,
        outputs: dict[str, Path],
        extra_meta: Optional[dict[str, Any]] = None,
    ) -> Path:
        """Write a manifest JSON documenting this ingestion run.

        Args:
            outputs: dict of dataset_name -> output file path
            extra_meta: additional metadata to include

        Returns:
            Path to the manifest file
        """
        manifest = {
            "source": self.source_name,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "files": {},
        }

        for name, path in outputs.items():
            if path.exists():
                file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
                manifest["files"][name] = {
                    "path": str(path),
                    "sha256": file_hash,
                    "size_bytes": path.stat().st_size,
                }

        if extra_meta:
            manifest["metadata"] = extra_meta

        manifest_path = self.manifest_dir / f"{self.source_name}_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

        logger.info(
            "[%s] Manifest written: %s",
            self.source_name.upper(), manifest_path,
        )
        return manifest_path
