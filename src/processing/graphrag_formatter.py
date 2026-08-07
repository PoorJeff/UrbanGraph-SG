"""GraphRAG formatter: convert processed entities to GraphRAG input format.

Generates:
- Entity description CSV (for GraphRAG entity extraction)
- Relationship CSV (deterministic relationships from rules)
- Community text files (one per planning area, for context)

Output format follows Microsoft GraphRAG conventions:
- entities.parquet: id, type, description, human_readable_id, ...
- relationships.parquet: source, target, description, weight, ...
"""

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import config

logger = logging.getLogger(__name__)


def format_all(
    entities_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """Convert all processed entities to GraphRAG format.

    Args:
        entities_dir: Directory with entity parquet files
        output_dir: GraphRAG input directory

    Returns:
        dict with paths to generated files
    """
    if entities_dir is None:
        entities_dir = config.data_dir / "processed" / "entities"
    if output_dir is None:
        output_dir = config.data_dir / "graphrag" / "input"

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    # Load all entity tables
    all_entities: list[dict[str, Any]] = []
    for efile in entities_dir.glob("*.parquet"):
        etype = efile.stem
        df = pd.read_parquet(efile)
        logger.info("Loading %s: %d entities", etype, len(df))
        for _, row in df.iterrows():
            entity = _entity_to_graphrag(row, etype)
            if entity:
                all_entities.append(entity)

    # Entity table
    entity_df = pd.DataFrame(all_entities)
    entity_path = output_dir / "entities.parquet"
    entity_df.to_parquet(entity_path, index=False)
    outputs["entities"] = entity_path
    logger.info("GraphRAG entities: %d total saved to %s", len(entity_df), entity_path)

    # Relationship table (deterministic rules only)
    rel_df = _build_relationships(entities_dir)
    rel_path = output_dir / "relationships.parquet"
    rel_df.to_parquet(rel_path, index=False)
    outputs["relationships"] = rel_path
    logger.info("GraphRAG relationships: %d total saved to %s", len(rel_df), rel_path)

    # Community text files (per planning area)
    community_dir = config.data_dir / "processed" / "community_texts"
    community_dir.mkdir(parents=True, exist_ok=True)
    _generate_community_texts(all_entities, community_dir)
    outputs["community_texts"] = community_dir

    logger.info("GraphRAG formatting complete: %d outputs", len(outputs))
    return outputs


def _entity_to_graphrag(row: Any, etype: str) -> dict[str, Any] | None:
    """Convert a single entity row to GraphRAG format.

    GraphRAG entity format:
    - id: unique identifier
    - type: entity type
    - description: natural language description for LLM
    - human_readable_id: display name
    - Additional metadata as properties
    """
    entity_id = str(row.get("entity_id", row.get("name", "")))
    if not entity_id:
        return None

    name = str(row.get("name", ""))
    entity_type = str(row.get("type", etype))

    # Build natural language description
    description_parts = [name]

    if entity_type in ("bus_stop", "mrt_station"):
        lines = row.get("lines", [])
        if isinstance(lines, list) and lines:
            description_parts.append(f"serving lines {', '.join(lines)}")
        pa = row.get("planning_area", "")
        if pa:
            description_parts.append(f"located in {pa} planning area")
        lat = row.get("lat")
        lon = row.get("lon")
        if lat and lon:
            description_parts.append(f"at coordinates ({lat:.4f}, {lon:.4f})")

    elif entity_type == "planning_area":
        pop = row.get("population", 0)
        if pop:
            description_parts.append(f"with population {pop:,}")
        region = row.get("region", "")
        if region:
            description_parts.append(f"in the {region} region of Singapore")

    elif entity_type == "hdb_town":
        price = row.get("avg_resale_price", 0)
        if price:
            description_parts.append(f"average HDB resale price S${price:,.0f}")
        count = row.get("transaction_count", 0)
        if count:
            description_parts.append(f"based on {count:,} transactions")

    elif entity_type == "weather_station":
        sid = row.get("station_id", "")
        if sid:
            description_parts.append(f"station ID {sid}")

    elif entity_type == "holiday":
        date = row.get("date", "")
        if date:
            description_parts.append(f"on {date}")
        subtype = row.get("subtype", "")
        if subtype:
            description_parts.append(f"({subtype} holiday)")

    description = ". ".join(description_parts) + "."

    result = {
        "id": entity_id,
        "type": entity_type,
        "name": name,
        "description": description,
        "human_readable_id": name,
    }

    # Add location data if present
    for key in ["lat", "lon", "latitude", "longitude", "planning_area"]:
        val = row.get(key)
        if val is not None and key in ["lat", "lon", "latitude", "longitude"]:
            try:
                result[key] = float(val)
            except (ValueError, TypeError):
                pass
        elif val:
            result[key] = str(val)

    return result


def _build_relationships(entities_dir: Path) -> pd.DataFrame:
    """Build deterministic relationships from entity data.

    Deterministic relationships (100% confidence, rule-based):
    - CONNECTS_TO: MRT stations on same line → neighbor stations
    - LOCATED_IN: TransportNode → PlanningArea
    - NEAR: bus stops ↔ MRT stations within 500m (simplified for MVP)
    - RECORDS: WeatherStation → WeatherEvent (handled in GraphRAG stage)
    """
    relationships: list[dict[str, Any]] = []

    # 1. CONNECTS_TO from MRT line order
    mrt_path = entities_dir / "transportnode.parquet"
    if mrt_path.exists():
        mrt_df = pd.read_parquet(mrt_path)
        mrt_only = mrt_df[mrt_df["subtype"] == "mrt"].copy()

        # Build CONNECTS_TO by ordering stations on the same line
        # Extract all unique line names from the list column (handles numpy arrays from parquet)
        all_lines: set[str] = set()
        for lines_val in mrt_only["lines"].dropna():
            vals = _to_list(lines_val)
            for v in vals:
                if v:
                    all_lines.add(str(v))
        logger.info("Found %d unique MRT lines: %s", len(all_lines), sorted(all_lines))
        for line_name in sorted(all_lines):
            _add_line_connections(mrt_only, line_name, relationships)

    # 2. LOCATED_IN
    for efile in entities_dir.glob("*.parquet"):
        df = pd.read_parquet(efile)
        if "planning_area" not in df.columns or "entity_id" not in df.columns:
            continue
        for _, row in df.iterrows():
            pa = row.get("planning_area", "")
            eid = row["entity_id"]
            if pa and eid and isinstance(pa, str):
                # Find planning area entity
                pa_slug = pa.lower().replace(" ", "-")
                relationships.append({
                    "source": str(eid),
                    "target": f"ura-area-{pa_slug}",
                    "relation": "LOCATED_IN",
                    "description": f"{row.get('name', eid)} is located in {pa}",
                    "weight": 1.0,
                    "generation": "rule",
                })

    # 3. NEAR: simplified — no distance calculation for MVP
    # Full implementation would compute haversine distances
    # For MVP, relationships near CBD/Orchard area are pre-computed

    rel_df = pd.DataFrame(relationships)
    # Deduplicate
    if not rel_df.empty:
        rel_df = rel_df.drop_duplicates(subset=["source", "target", "relation"])
    logger.info("Built %d deterministic relationships", len(rel_df))
    return rel_df


def _to_list(val):
    """Convert numpy array or list to a plain Python list."""
    import numpy as np
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, (list, tuple)):
        return list(val)
    if isinstance(val, str) and val:
        return [val]
    return []


def _add_line_connections(
    mrt_df: pd.DataFrame,
    line_name: str,
    relationships: list[dict[str, Any]],
) -> None:
    """Add CONNECTS_TO relationships for stations on a given MRT line.

    MRT station codes follow the pattern: XX## (e.g., EW24, NS17).
    We sort by the numeric part of the code to determine order on the line.
    """
    import re

    # Filter stations that have this line
    def has_line(lines_val, target_line):
        vals = _to_list(lines_val)
        return target_line in vals

    line_stations = mrt_df[mrt_df["lines"].apply(
        lambda x: has_line(x, line_name)
    )].copy()

    if len(line_stations) < 2:
        return

    # Sort by station code numeric part
    def extract_number(code):
        if isinstance(code, str):
            nums = re.findall(r"\d+", code)
            return int(nums[0]) if nums else 0
        return 0

    line_stations["_sort_key"] = line_stations["code"].apply(extract_number)
    line_stations = line_stations.sort_values("_sort_key")

    # Create consecutive pairs
    stations_list = line_stations.to_dict("records")
    for i in range(len(stations_list) - 1):
        s1 = stations_list[i]
        s2 = stations_list[i + 1]
        relationships.append({
            "source": s1["entity_id"],
            "target": s2["entity_id"],
            "relation": "CONNECTS_TO",
            "description": f"{s1['name']} connects to {s2['name']} on {line_name} line",
            "weight": 1.0,
            "line": line_name,
            "generation": "rule",
        })


def _generate_community_texts(
    all_entities: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    """Generate natural language community text files.

    One file per planning area, describing all entities in that area.
    These serve as input for GraphRAG entity extraction.
    """
    from collections import defaultdict

    # Group entities by planning_area
    by_area: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entity in all_entities:
        pa = entity.get("planning_area", "")
        if pa:
            by_area[pa].append(entity)

    for area_name, ents in sorted(by_area.items()):
        if len(ents) < 2:
            continue

        area_slug = area_name.lower().replace(" ", "_")
        path = output_dir / f"{area_slug}.txt"

        lines = [f"# {area_name} Planning Area\n"]

        # Population info
        pop_val = next((e for e in ents if e.get("type") == "planning_area"), None)
        if pop_val:
            p = pop_val.get("population", "N/A")
            if isinstance(p, (int, float)):
                lines.append(f"Population: {int(p):,}\n\n")
            else:
                lines.append(f"Population: {p}\n\n")

        # Transport nodes
        transport = [e for e in ents if e.get("type") in ("bus_stop", "mrt_station")]
        if transport:
            lines.append("## Transport Nodes\n")
            for t in transport[:20]:  # Cap per area
                lines.append(f"- {t['name']}: {t.get('description', '')}\n")

        # HDB
        hdb = [e for e in ents if e.get("type") == "hdb_town"]
        if hdb:
            lines.append("\n## Housing\n")
            for h in hdb:
                lines.append(f"- {h['name']}: {h.get('description', '')}\n")

        path.write_text("".join(lines), encoding="utf-8")

    logger.info(
        "Generated %d community text files in %s",
        len(by_area), output_dir,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=getattr(logging, config.settings.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    format_all()
