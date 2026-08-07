"""Run all ingestion agents sequentially.

Usage:
    python -m src.ingestion.ingest_all
    make ingest-all

This script runs all 6 ingestion agents and reports results.
Individual agents can also be run independently:
    python -m src.ingestion.lta
    python -m src.ingestion.nea
    etc.
"""

import logging
import sys
import time
from pathlib import Path

from src.config import config
from src.ingestion.calendar import CalendarIngestionAgent
from src.ingestion.hdb import HDBIngestionAgent
from src.ingestion.lta import LTAIngestionAgent
from src.ingestion.nea import NEAWeatherIngestionAgent
from src.ingestion.onemap import OneMapIngestionAgent
from src.ingestion.singstat import SingStatIngestionAgent

logger = logging.getLogger(__name__)


def run_all():
    """Run all ingestion agents and report results."""
    start_time = time.time()

    agents = [
        ("Calendar", CalendarIngestionAgent),
        ("LTA", LTAIngestionAgent),
        ("OneMap", OneMapIngestionAgent),
        ("NEA Weather", NEAWeatherIngestionAgent),
        ("HDB", HDBIngestionAgent),
        ("SingStat", SingStatIngestionAgent),
    ]

    results: dict[str, bool] = {}
    total_files = 0

    for name, agent_cls in agents:
        logger.info("=" * 60)
        logger.info("Running %s ingestion agent...", name)
        logger.info("=" * 60)
        try:
            agent = agent_cls()
            outputs = agent.ingest()
            results[name] = True
            total_files += len(outputs)
            logger.info("[%s] SUCCESS: %d files produced", name, len(outputs))
        except Exception as e:
            logger.error("[%s] FAILED: %s", name, e)
            results[name] = False

    elapsed = time.time() - start_time

    # Summary
    logger.info("=" * 60)
    logger.info("INGESTION SUMMARY")
    logger.info("=" * 60)
    for name, success in results.items():
        status = "PASS" if success else "FAIL"
        logger.info("  %-20s %s", name, status)
    logger.info("-" * 60)
    logger.info("  Total files: %d", total_files)
    logger.info("  Elapsed: %.1f seconds", elapsed)
    logger.info("=" * 60)

    # Report failures
    failed = [name for name, ok in results.items() if not ok]
    if failed:
        logger.warning("Failed agents: %s", ", ".join(failed))

    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=getattr(logging, config.settings.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run_all()
