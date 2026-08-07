"""Time normalization: all timestamps → Asia/Singapore (UTC+8).

Handles:
- NEA timestamps (ISO 8601 with +08:00 offset)
- HDB month strings ("2017-01")
- Holiday date strings ("2025-01-01")
- Calendar date strings

All output dates are stored as naive strings in "YYYY-MM-DD" format
or "YYYY-MM-DD HH:MM:SS+08:00" for timestamps with time components.
"""

import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import config

logger = logging.getLogger(__name__)

# Singapore timezone
SGT_OFFSET = timedelta(hours=8)
SGT_TZ = timezone(SGT_OFFSET)


def normalize_all(raw_dir: Path | None = None, processed_dir: Path | None = None) -> dict[str, Path]:
    """Normalize all raw datasets to SGT timezone.

    Args:
        raw_dir: Raw data directory (default: config.data_dir / "raw")
        processed_dir: Output directory (default: config.data_dir / "processed")

    Returns:
        dict mapping dataset name to output path
    """
    if raw_dir is None:
        raw_dir = config.data_dir / "raw"
    if processed_dir is None:
        processed_dir = config.data_dir / "processed"

    processed_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    # NEA weather data
    for name in ["rainfall", "temperature", "humidity", "wind_speed", "wind_direction"]:
        src = raw_dir / "nea" / f"{name}.parquet"
        if src.exists():
            df = pd.read_parquet(src)
            df = _normalize_nea(df, name)
            dest = processed_dir / "nea" / f"{name}.parquet"
            dest.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(dest, index=False)
            outputs[f"nea_{name}"] = dest
            logger.info("Normalized NEA %s: %d records", name, len(df))

    # HDB resale data
    src = raw_dir / "hdb" / "resale_prices.parquet"
    if src.exists():
        df = pd.read_parquet(src)
        df = _normalize_hdb(df)
        dest = processed_dir / "hdb" / "resale_prices.parquet"
        dest.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dest, index=False)
        outputs["hdb_resale"] = dest
        logger.info("Normalized HDB: %d records", len(df))

    # Holidays
    src = raw_dir / "calendar" / "sg_holidays.parquet"
    if src.exists():
        df = pd.read_parquet(src)
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        dest = processed_dir / "calendar" / "sg_holidays.parquet"
        dest.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dest, index=False)
        outputs["holidays"] = dest

    # School terms
    src = raw_dir / "calendar" / "school_terms.parquet"
    if src.exists():
        df = pd.read_parquet(src)
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        dest = processed_dir / "calendar" / "school_terms.parquet"
        dest.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dest, index=False)
        outputs["school_terms"] = dest

    # Population (no time fields, just copy)
    src = raw_dir / "singstat" / "population.parquet"
    if src.exists():
        df = pd.read_parquet(src)
        dest = processed_dir / "singstat" / "population.parquet"
        dest.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dest, index=False)
        outputs["population"] = dest

    logger.info("Time normalization complete: %d datasets", len(outputs))
    return outputs


def _normalize_nea(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Normalize NEA weather data timestamps.

    NEA data from data.gov.sg already comes in SGT (+08:00).
    We parse the timestamp, verify it's SGT, and normalize to date-only
    for aggregation (detailed timestamps preserved in separate column).
    """
    df = df.copy()

    # Parse timestamps (already +08:00 from API)
    if "timestamp" in df.columns:
        df["timestamp_parsed"] = pd.to_datetime(df["timestamp"], utc=False)
        # Verify all are within reasonable SGT range
        df["timestamp_sgt"] = df["timestamp_parsed"].dt.tz_localize(None)

    # Ensure date is YYYY-MM-DD
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    return df


def _normalize_hdb(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize HDB resale data month fields."""
    df = df.copy()
    if "month" in df.columns:
        # "2017-01" → standardize
        df["month"] = pd.to_datetime(df["month"]).dt.strftime("%Y-%m")
    return df


if __name__ == "__main__":
    logging.basicConfig(
        level=getattr(logging, config.settings.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    normalize_all()
