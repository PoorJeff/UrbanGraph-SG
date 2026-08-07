"""Calendar ingestion agent: Singapore public holidays and school terms.

Uses the `holidays` Python package for public holidays (no API key needed).
School terms are generated from MOE's published calendar pattern.
"""

import logging
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import config
from src.ingestion.base import BaseIngestionAgent

logger = logging.getLogger(__name__)

# Singapore school terms (approximate, based on MOE calendar pattern)
# Format: (term, start_mmdd, end_mmdd)
# These are typical dates; exact dates vary slightly each year
SCHOOL_TERMS_2025 = [
    (1, "2025-01-02", "2025-03-14"),
    (2, "2025-03-24", "2025-05-30"),
    (3, "2025-06-30", "2025-09-05"),
    (4, "2025-09-15", "2025-11-21"),
]

SCHOOL_TERMS_2026 = [
    (1, "2026-01-02", "2026-03-13"),
    (2, "2026-03-23", "2026-05-29"),
    (3, "2026-06-29", "2026-09-04"),
    (4, "2026-09-14", "2026-11-20"),
]

# School holidays between terms (used for school_holiday flag)
SCHOOL_HOLIDAYS_2025 = [
    ("2025-03-15", "2025-03-23"),
    ("2025-05-31", "2025-06-29"),
    ("2025-09-06", "2025-09-14"),
    ("2025-11-22", "2025-12-31"),
]
SCHOOL_HOLIDAYS_2026 = [
    ("2026-03-14", "2026-03-22"),
    ("2026-05-30", "2026-06-28"),
    ("2026-09-05", "2026-09-13"),
    ("2026-11-21", "2026-12-31"),
]


class CalendarIngestionAgent(BaseIngestionAgent):
    """Ingest Singapore public holidays and school term calendar."""

    source_name = "calendar"

    def ingest(self) -> dict[str, Path]:
        """Generate holiday and school term data.

        Returns:
            dict with holiday_path and school_term_path
        """
        logger.info("[CALENDAR] Generating Singapore calendar data...")

        start_year = self.settings.data_start_year
        end_year = self.settings.data_end_year

        holidays_df = self._generate_holidays(start_year, end_year)
        school_df = self._generate_school_calendar(start_year, end_year)

        outputs: dict[str, Path] = {}

        holiday_path = self.save_dataframe(
            holidays_df, "sg_holidays.parquet", "holidays"
        )
        outputs["holidays"] = holiday_path

        school_path = self.save_dataframe(
            school_df, "school_terms.parquet", "school_terms"
        )
        outputs["school_terms"] = school_path

        self.write_manifest(
            outputs,
            extra_meta={
                "years": f"{start_year}-{end_year}",
                "holiday_count": len(holidays_df),
                "school_term_days": len(school_df),
            },
        )

        logger.info(
            "[CALENDAR] Complete: %d holidays, %d school days",
            len(holidays_df), len(school_df),
        )
        return outputs

    def _generate_holidays(self, start_year: int, end_year: int) -> pd.DataFrame:
        """Generate public holidays using the holidays package."""
        try:
            import holidays
        except ImportError:
            logger.warning(
                "[CALENDAR] 'holidays' package not installed. "
                "Install with: pip install holidays"
            )
            return pd.DataFrame(
                columns=["date", "name", "type"]
            )

        records: list[dict[str, Any]] = []
        sg_holidays = holidays.Singapore(years=range(start_year, end_year + 1))

        for dt, name in sorted(sg_holidays.items()):
            records.append({
                "date": dt.isoformat(),
                "name": name,
                "type": "public",
            })

        logger.info(
            "[CALENDAR] Found %d public holidays (%d-%d)",
            len(records), start_year, end_year,
        )
        return pd.DataFrame(records)

    def _generate_school_calendar(
        self, start_year: int, end_year: int
    ) -> pd.DataFrame:
        """Generate school term and holiday calendar."""
        terms_map = {2025: SCHOOL_TERMS_2025, 2026: SCHOOL_TERMS_2026}
        holidays_map = {2025: SCHOOL_HOLIDAYS_2025, 2026: SCHOOL_HOLIDAYS_2026}

        records: list[dict[str, Any]] = []

        for year in range(start_year, end_year + 1):
            terms = terms_map.get(year, [])
            holidays = holidays_map.get(year, [])

            # Generate individual dates for each term
            for term_num, start_str, end_str in terms:
                start = date.fromisoformat(start_str)
                end = date.fromisoformat(end_str)
                current = start
                while current <= end:
                    # Check if this is a weekend
                    is_weekend = current.weekday() >= 5
                    if not is_weekend:
                        records.append({
                            "date": current.isoformat(),
                            "year": year,
                            "term": term_num,
                            "is_school_day": True,
                            "type": "school_term",
                        })
                    current = date.fromordinal(current.toordinal() + 1)

            # Generate school holiday dates
            for start_str, end_str in holidays:
                start = date.fromisoformat(start_str)
                end = date.fromisoformat(end_str)
                current = start
                while current <= end:
                    records.append({
                        "date": current.isoformat(),
                        "year": year,
                        "term": 0,
                        "is_school_day": False,
                        "type": "school_holiday",
                    })
                    current = date.fromordinal(current.toordinal() + 1)

        logger.info(
            "[CALENDAR] Generated %d school calendar entries (%d-%d)",
            len(records), start_year, end_year,
        )
        return pd.DataFrame(records)


if __name__ == "__main__":
    logging.basicConfig(
        level=getattr(logging, config.settings.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    agent = CalendarIngestionAgent()
    agent.ingest()
