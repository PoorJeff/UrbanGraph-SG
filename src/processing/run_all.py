"""Main entry point for the data processing pipeline.

Orchestrates: time normalization → spatial validation → entity resolution → GraphRAG formatting → quality report.
Triggered by: `make process`
"""

import logging
import time
from pathlib import Path

from src.config import config

logger = logging.getLogger(__name__)


def run_processing_pipeline() -> dict[str, bool]:
    """Execute the full data processing pipeline.

    Steps:
    1. Time normalization (all timestamps → Asia/Singapore)
    2. Spatial validation (coordinates within Singapore bounds)
    3. Entity resolution (merge duplicate entities across sources)
    4. GraphRAG formatting (generate entity/relationship CSVs)
    5. Quality report generation

    Returns:
        dict with step_name -> success boolean
    """
    results: dict[str, bool] = {}
    pipeline_start = time.time()

    logger.info("=" * 60)
    logger.info("UrbanGraph-SG Data Processing Pipeline")
    logger.info("=" * 60)

    raw_dir = config.data_dir / "raw"
    processed_dir = config.data_dir / "processed"

    # Step 1: Time normalization
    step = "[1/5] Time normalization"
    logger.info(step)
    try:
        from src.processing.time_normalizer import normalize_all
        normalize_all(raw_dir, processed_dir)
        results["time_normalization"] = True
        logger.info("%s: DONE", step)
    except Exception as e:
        logger.error("%s: FAILED — %s", step, e)
        results["time_normalization"] = False

    # Step 2: Spatial validation
    step = "[2/5] Spatial validation"
    logger.info(step)
    try:
        from src.processing.spatial_validator import validate_all
        result = validate_all(raw_dir, processed_dir)
        anomaly_count = sum(len(v) for v in result.get("anomalies", {}).values())
        results["spatial_validation"] = True
        logger.info("%s: DONE (%d anomalies)", step, anomaly_count)
    except Exception as e:
        logger.error("%s: FAILED — %s", step, e)
        results["spatial_validation"] = False

    # Step 3: Entity resolution
    step = "[3/5] Entity resolution"
    logger.info(step)
    try:
        from src.processing.entity_resolution import resolve_all
        entities = resolve_all(processed_dir)
        total = sum(len(df) for df in entities.values())
        results["entity_resolution"] = True
        logger.info(
            "%s: DONE (%d entities across %d types)",
            step, total, len(entities),
        )
    except Exception as e:
        logger.error("%s: FAILED — %s", step, e)
        results["entity_resolution"] = False

    # Step 4: GraphRAG formatting
    step = "[4/5] GraphRAG formatting"
    logger.info(step)
    try:
        from src.processing.graphrag_formatter import format_all
        format_all()
        results["graphrag_formatting"] = True
        logger.info("%s: DONE", step)
    except Exception as e:
        logger.error("%s: FAILED — %s", step, e)
        results["graphrag_formatting"] = False

    # Step 5: Quality report
    step = "[5/5] Quality report"
    logger.info(step)
    try:
        from src.processing.quality_reporter import generate_report
        report = generate_report(raw_dir, processed_dir)
        summary = report.get("summary", {})
        results["quality_report"] = True
        logger.info(
            "%s: DONE — %d entities, %d anomalies",
            step,
            summary.get("total_entities", 0),
            summary.get("anomaly_count", 0),
        )
    except Exception as e:
        logger.error("%s: FAILED — %s", step, e)
        results["quality_report"] = False

    elapsed = time.time() - pipeline_start

    # Summary
    logger.info("=" * 60)
    logger.info("PIPELINE SUMMARY (%.1fs)", elapsed)
    logger.info("=" * 60)
    for step_name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        logger.info("  %-25s %s", step_name, status)
    passed = sum(1 for v in results.values() if v)
    logger.info("  %d/%d steps passed", passed, len(results))
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=getattr(logging, config.settings.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run_processing_pipeline()
