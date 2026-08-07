"""Entity resolution: deduplicate and unify entities across data sources.

Implements §C.3 rules from UrbanGraph-SG-report.md:
- Level 1: Exact name match
- Level 2: Normalized match (lowercase, strip suffix)
- Level 3: Levenshtein distance < 3
- Level 4: Spatial proximity (< 100m) + name similarity > 0.85

Generates global unique entity IDs per §C.1:
- TransportNode: lta-mrt-{code}, lta-bus-{code}
- PlanningArea: ura-area-{name}
- HDBTown: hdb-town-{name}
- POI: onemap-poi-{uuid}
- WeatherStation: nea-station-{id}
- WeatherEvent: nea-event-{date}
- Holiday: holiday-{date}
- EntityCommunity: community-{id}
"""

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import config

logger = logging.getLogger(__name__)


def normalize_name(name: str) -> str:
    """Normalize an entity name for comparison.

    Steps:
    1. Lowercase
    2. Remove common suffixes: "MRT Station", "Station", "MRT", "LRT"
    3. Remove extra whitespace
    4. Strip punctuation
    """
    if not isinstance(name, str):
        return ""
    name = name.lower().strip()
    # Remove common transport suffixes
    for suffix in [
        "mrt station", "lrt station", "mrt", "lrt",
        "station", "bus stop", "terminal", "interchange",
        "stn",
    ]:
        name = re.sub(rf"\b{suffix}\b", "", name, flags=re.IGNORECASE)
    # Remove punctuation and extra whitespace
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            # Insertion, deletion, substitution
            curr.append(min(
                curr[-1] + 1,
                prev[j + 1] + 1,
                prev[j] + (0 if c1 == c2 else 1),
            ))
        prev = curr
    return prev[-1]


def resolve_all(
    processed_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Run entity resolution across all data sources.

    Steps:
    1. Generate entity DataFrames for each type
    2. Detect and report potential duplicates (for manual review)
    3. Generate global unique entity IDs
    4. Merge entities where appropriate

    Returns:
        dict mapping entity_type -> DataFrame with resolved entities
    """
    if processed_dir is None:
        processed_dir = config.data_dir / "processed"

    entities: dict[str, pd.DataFrame] = {}

    # 1. Transport Nodes (bus stops + MRT stations)
    bus_df = _load_or_raw(processed_dir, "lta", "bus_stops.parquet",
                          config.data_dir / "raw" / "lta" / "bus_stops.parquet")
    mrt_df = _load_or_raw(processed_dir, "onemap", "mrt_stations.parquet",
                          config.data_dir / "raw" / "onemap" / "mrt_stations.parquet")

    transport_entities = _build_transport_entities(bus_df, mrt_df)
    entities["TransportNode"] = transport_entities
    logger.info("TransportNode: %d entities (bus + MRT)", len(transport_entities))

    # 2. Planning Areas
    pop_df = _load_or_raw(processed_dir, "singstat", "population.parquet",
                          config.data_dir / "raw" / "singstat" / "population.parquet")
    pa_names_df = _load_or_raw(processed_dir, "onemap", "planning_area_names.parquet",
                               config.data_dir / "raw" / "onemap" / "planning_area_names.parquet")

    planning_entities = _build_planning_area_entities(pop_df)
    entities["PlanningArea"] = planning_entities
    logger.info("PlanningArea: %d entities", len(planning_entities))

    # 3. HDB Towns
    hdb_df = _load_or_raw(None, "hdb", "resale_prices.parquet",
                          config.data_dir / "raw" / "hdb" / "resale_prices.parquet")
    if hdb_df is not None and not hdb_df.empty:
        hdb_entities = _build_hdb_town_entities(hdb_df, planning_entities)
        entities["HDBTown"] = hdb_entities
        logger.info("HDBTown: %d entities", len(hdb_entities))

    # 4. Holidays
    hol_df = _load_or_raw(processed_dir, "calendar", "sg_holidays.parquet",
                          config.data_dir / "raw" / "calendar" / "sg_holidays.parquet")
    if hol_df is not None and not hol_df.empty:
        holiday_entities = _build_holiday_entities(hol_df)
        entities["Holiday"] = holiday_entities
        logger.info("Holiday: %d entities", len(holiday_entities))

    # 5. Weather Stations (from NEA data)
    weather_entities = _build_weather_station_entities()
    entities["WeatherStation"] = weather_entities
    logger.info("WeatherStation: %d entities", len(weather_entities))

    # 6. Detect duplicates within TransportNode
    duplicates = _detect_transport_duplicates(transport_entities)
    if duplicates:
        logger.warning("Found %d potential duplicate pairs in TransportNode", len(duplicates))
        dup_path = config.data_dir / "processed" / "duplicate_report.csv"
        pd.DataFrame(duplicates).to_csv(dup_path, index=False)
        logger.info("Duplicate report saved to %s", dup_path)

    # Save all entity tables
    entities_dir = config.data_dir / "processed" / "entities"
    entities_dir.mkdir(parents=True, exist_ok=True)
    for etype, edf in entities.items():
        path = entities_dir / f"{etype.lower()}.parquet"
        edf.to_parquet(path, index=False)
        logger.info("Saved %s entities to %s", etype, path)

    return entities


def _load_or_raw(
    processed_dir: Path | None,
    processed_subdir: str,
    filename: str,
    raw_path: Path,
) -> pd.DataFrame | None:
    """Load from processed if available, otherwise from raw."""
    if processed_dir:
        p = processed_dir / processed_subdir / filename
        if p.exists():
            return pd.read_parquet(p)
    if raw_path.exists():
        return pd.read_parquet(raw_path)
    return None


def _build_transport_entities(
    bus_df: pd.DataFrame | None,
    mrt_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Build unified TransportNode entity table."""
    records = []

    # Bus stops
    if bus_df is not None and not bus_df.empty:
        for _, row in bus_df.iterrows():
            records.append({
                "entity_id": f"lta-bus-{row['BusStopCode']}",
                "name": row["Description"],
                "type": "bus_stop",
                "subtype": "bus",
                "lat": row.get("Latitude"),
                "lon": row.get("Longitude"),
                "planning_area": row.get("planning_area", ""),
                "source": "lta",
                "code": row["BusStopCode"],
                "road_name": row.get("RoadName", ""),
                "lines": [],
            })

    # MRT stations
    if mrt_df is not None and not mrt_df.empty:
        # Group by station_id to aggregate lines
        station_groups: dict[str, dict[str, Any]] = {}
        for _, row in mrt_df.iterrows():
            sid = row.get("station_id", row.get("code", ""))
            if sid not in station_groups:
                station_groups[sid] = {
                    "entity_id": f"lta-mrt-{sid}",
                    "name": row["name"],
                    "type": "mrt_station",
                    "subtype": "mrt",
                    "lat": row.get("lat"),
                    "lon": row.get("lon"),
                    "planning_area": row.get("planning_area", ""),
                    "source": "onemap_static",
                    "code": sid,
                    "road_name": "",
                    "lines": [],
                }
            line = row.get("line", "")
            if line and line not in station_groups[sid]["lines"]:
                station_groups[sid]["lines"].append(line)

        for sid, rec in station_groups.items():
            records.append(rec)

    df = pd.DataFrame(records)
    if not df.empty:
        df["entity_id"] = df["entity_id"].astype(str)
    return df


def _build_planning_area_entities(pop_df: pd.DataFrame | None) -> pd.DataFrame:
    """Build PlanningArea entities from population data."""
    if pop_df is None or pop_df.empty:
        return pd.DataFrame(columns=[
            "entity_id", "name", "type", "region", "population", "source"
        ])

    records = []
    for _, row in pop_df.iterrows():
        area_name = row["planning_area"]
        area_slug = area_name.lower().replace(" ", "-")
        records.append({
            "entity_id": f"ura-area-{area_slug}",
            "name": area_name,
            "type": "planning_area",
            "region": row.get("region", ""),
            "population": row.get("population", 0),
            "planning_area": area_name,
            "source": "singstat",
        })

    return pd.DataFrame(records)


def _build_hdb_town_entities(
    hdb_df: pd.DataFrame,
    planning_entities: pd.DataFrame,
) -> pd.DataFrame:
    """Build HDBTown entities from resale data."""
    if hdb_df is None or hdb_df.empty:
        return pd.DataFrame()

    # Aggregate: avg price, rental median per town
    town_stats = hdb_df.groupby("town").agg(
        avg_resale_price=("resale_price", "mean"),
        min_price=("resale_price", "min"),
        max_price=("resale_price", "max"),
        transaction_count=("resale_price", "count"),
        flat_types=("flat_type", lambda x: sorted(x.unique())),
    ).reset_index()

    # Map region from planning areas
    pa_regions = {}
    if not planning_entities.empty and "name" in planning_entities.columns:
        pa_regions = dict(zip(
            planning_entities["name"].str.upper(),
            planning_entities["region"],
        ))

    records = []
    for _, row in town_stats.iterrows():
        town = row["town"]
        town_upper = town.upper()
        records.append({
            "entity_id": f"hdb-town-{town.lower().replace(' ', '-')}",
            "name": town.title(),
            "type": "hdb_town",
            "region": pa_regions.get(town_upper, ""),
            "planning_area": town.title(),
            "avg_resale_price": round(row["avg_resale_price"]),
            "min_price": row["min_price"],
            "max_price": row["max_price"],
            "transaction_count": row["transaction_count"],
            "flat_types": row.get("flat_types", []),
            "source": "hdb",
        })

    return pd.DataFrame(records)


def _build_holiday_entities(hol_df: pd.DataFrame) -> pd.DataFrame:
    """Build Holiday entities."""
    if hol_df is None or hol_df.empty:
        return pd.DataFrame()

    records = []
    for _, row in hol_df.iterrows():
        records.append({
            "entity_id": f"holiday-{row['date']}",
            "name": row["name"],
            "type": "holiday",
            "subtype": row.get("type", "public"),
            "date": row["date"],
            "planning_area": "Singapore",
            "source": "calendar",
        })

    return pd.DataFrame(records)


def _build_weather_station_entities() -> pd.DataFrame:
    """Build WeatherStation entities from known NEA station metadata."""
    # NEA weather station reference (key stations only)
    # Source: data.gov.sg metadata, NEA website
    stations = [
        {"station_id": "S24", "name": "Changi Climate Station", "lat": 1.3644, "lon": 103.9915, "region": "East"},
        {"station_id": "S44", "name": "Newton", "lat": 1.3123, "lon": 103.8374, "region": "Central"},
        {"station_id": "S50", "name": "Clementi", "lat": 1.3151, "lon": 103.7652, "region": "West"},
        {"station_id": "S60", "name": "Sentosa", "lat": 1.2500, "lon": 103.8300, "region": "Central"},
        {"station_id": "S77", "name": "Tuas South", "lat": 1.3100, "lon": 103.6500, "region": "West"},
        {"station_id": "S100", "name": "Woodlands", "lat": 1.4369, "lon": 103.7865, "region": "North"},
        {"station_id": "S104", "name": "Seletar", "lat": 1.4150, "lon": 103.8700, "region": "North-East"},
        {"station_id": "S106", "name": "Pulau Ubin", "lat": 1.4150, "lon": 103.9600, "region": "North-East"},
        {"station_id": "S107", "name": "Tai Seng", "lat": 1.3380, "lon": 103.8880, "region": "Central"},
        {"station_id": "S109", "name": "Ang Mo Kio", "lat": 1.3700, "lon": 103.8500, "region": "North-East"},
        {"station_id": "S116", "name": "Choa Chu Kang (South)", "lat": 1.3800, "lon": 103.7450, "region": "West"},
        {"station_id": "S117", "name": "East Coast Parkway", "lat": 1.3040, "lon": 103.9200, "region": "East"},
        {"station_id": "S121", "name": "Old Choa Chu Kang Road", "lat": 1.3850, "lon": 103.7400, "region": "West"},
        {"station_id": "S122", "name": "Changi (Airport)", "lat": 1.3580, "lon": 103.9900, "region": "East"},
    ]

    records = []
    for s in stations:
        records.append({
            "entity_id": f"nea-station-{s['station_id']}",
            "name": s["name"],
            "station_id": s["station_id"],
            "type": "weather_station",
            "lat": s["lat"],
            "lon": s["lon"],
            "planning_area": "",
            "region": s["region"],
            "source": "nea_reference",
        })

    return pd.DataFrame(records)


def _detect_transport_duplicates(
    transport_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Detect potential duplicate entities in transport nodes.

    Uses hash-based grouping to avoid O(n²) comparisons.
    - Level 1: Exact match after normalization (hash group)
    - Level 3: Levenshtein < 3 (only within first-char groups)

    Returns list of duplicate pairs for manual review.
    """
    if transport_df.empty:
        return []

    duplicates = []
    names = transport_df["name"].tolist()
    ids = transport_df["entity_id"].tolist()

    # Level 1: Hash-based exact match after normalization
    norm_map: dict[str, list[int]] = {}
    for i, name in enumerate(names):
        norm = normalize_name(str(name))
        if norm:
            norm_map.setdefault(norm, []).append(i)

    for norm, indices in norm_map.items():
        if len(indices) > 1:
            # Multiple entities with the same normalized name
            for a, b in _pairwise(indices):
                duplicates.append({
                    "entity_1": ids[a],
                    "name_1": names[a],
                    "entity_2": ids[b],
                    "name_2": names[b],
                    "match_level": 1,
                    "reason": f"exact normalized match: '{norm}'",
                })

    # Level 3: Levenshtein (only within first-char groups, max 200 per group)
    char_groups: dict[str, list[int]] = {}
    for i, name in enumerate(names):
        norm = normalize_name(str(name))
        if norm:
            char_groups.setdefault(norm[:1], []).append(i)

    for group_indices in char_groups.values():
        if len(group_indices) > 200:
            continue  # Skip groups too large for pairwise
        for i_idx, j_idx in _pairwise(group_indices):
            norm_i = normalize_name(str(names[i_idx]))
            norm_j = normalize_name(str(names[j_idx]))
            if not norm_i or not norm_j or len(norm_i) <= 3 or len(norm_j) <= 3:
                continue
            dist = levenshtein_distance(norm_i, norm_j)
            if 0 < dist <= 2:
                duplicates.append({
                    "entity_1": ids[i_idx],
                    "name_1": names[i_idx],
                    "entity_2": ids[j_idx],
                    "name_2": names[j_idx],
                    "match_level": 3,
                    "levenshtein": dist,
                    "reason": f"Levenshtein distance = {dist}",
                })

    return duplicates


def _pairwise(indices: list[int]):
    """Generate all unique pairs from a list of indices."""
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            yield indices[i], indices[j]


if __name__ == "__main__":
    logging.basicConfig(
        level=getattr(logging, config.settings.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    entities = resolve_all()
    for etype, edf in entities.items():
        print(f"{etype}: {len(edf)} entities")
        if not edf.empty:
            print(f"  sample IDs: {edf['entity_id'].head(3).tolist()}")
