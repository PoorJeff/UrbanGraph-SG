"""Neo4j data loader.

Loads all entities and relationships from the GraphRAG pipeline
into Neo4j in batches for performance.

Steps:
1. Create schema (constraints, indexes)
2. Load entities by type
3. Load relationships
4. Load community structure
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import config
from src.graph.neo4j_client import batch_execute, run_query
from src.graph.schema import create_constraints_and_indexes, get_schema_summary

logger = logging.getLogger(__name__)


def load_all(
    entities_path: Path | None = None,
    relationships_path: Path | None = None,
    communities_path: Path | None = None,
    entity_community_map_path: Path | None = None,
) -> dict[str, int]:
    """Load all data into Neo4j.

    Returns:
        dict with node and relationship counts
    """
    if entities_path is None:
        entities_path = config.data_dir / "graphrag" / "input" / "entities.parquet"
    if relationships_path is None:
        relationships_path = config.data_dir / "graphrag" / "input" / "relationships.parquet"
    if communities_path is None:
        communities_path = config.data_dir / "graphrag" / "output" / "communities.parquet"
    if entity_community_map_path is None:
        entity_community_map_path = config.data_dir / "graphrag" / "output" / "entity_community_map.parquet"

    logger.info("=" * 60)
    logger.info("Loading knowledge graph into Neo4j")
    logger.info("=" * 60)

    # Step 0: Create schema
    create_constraints_and_indexes()

    # Step 1: Load entities
    entities_df = pd.read_parquet(entities_path)
    logger.info("Loading %d entities...", len(entities_df))
    node_count = _load_entities(entities_df)

    # Step 2: Load relationships
    if relationships_path.exists():
        rels_df = pd.read_parquet(relationships_path)
        logger.info("Loading %d relationships...", len(rels_df))
        rel_count = _load_relationships(rels_df)
    else:
        rel_count = 0

    # Step 3: Load community structure
    if communities_path.exists() and entity_community_map_path.exists():
        logger.info("Loading community structure...")
        _load_communities(communities_path, entity_community_map_path)

    # Verify
    summary = get_schema_summary()
    logger.info("=" * 60)
    logger.info("Neo4j Load Complete:")
    for label, count in summary.items():
        logger.info("  %-20s %d", label, count)
    logger.info("=" * 60)

    return summary


def _load_entities(df: pd.DataFrame) -> int:
    """Load entities into Neo4j nodes.

    Maps entity types to Neo4j labels:
    - bus_stop / mrt_station → :TransportNode
    - planning_area → :PlanningArea
    - weather_station → :WeatherStation
    - holiday → :Holiday
    - hdb_town → :HDBTown
    """
    label_map = {
        "bus_stop": "TransportNode",
        "mrt_station": "TransportNode",
        "planning_area": "PlanningArea",
        "weather_station": "WeatherStation",
        "holiday": "Holiday",
        "hdb_town": "HDBTown",
    }

    queries: list[tuple[str, dict[str, Any]]] = []

    for _, row in df.iterrows():
        eid = str(row.get("id", ""))
        etype = str(row.get("type", ""))
        label = label_map.get(etype, "TransportNode")

        # Build properties
        props: dict[str, Any] = {"id": eid}

        # Standard fields
        for field in ["name", "lat", "lon", "planning_area", "description"]:
            val = row.get(field)
            if val is not None and not (isinstance(val, float) and pd.isna(val)):
                props[field] = val

        # Type-specific fields
        if etype == "mrt_station":
            props["transport_type"] = "mrt"
        elif etype == "bus_stop":
            props["transport_type"] = "bus"

        if etype == "planning_area":
            for f in ["population", "region"]:
                val = row.get(f)
                if val is not None:
                    props[f] = val

        if etype == "holiday":
            props["date"] = row.get("date", "")

        if etype == "hdb_town":
            for f in ["avg_resale_price", "min_price", "max_price", "transaction_count"]:
                val = row.get(f)
                if val is not None:
                    props[f] = val

        # Build CREATE or MERGE query
        prop_str = ", ".join(
            f"n.{k} = ${k}" for k in props
        )
        params = {k: v for k, v in props.items()}

        query = f"MERGE (n:{label} {{id: $id}}) SET {prop_str}"
        queries.append((query, params))

    count = batch_execute(queries)
    logger.info("Loaded %d nodes", count)
    return count


def _load_relationships(df: pd.DataFrame) -> int:
    """Load relationships into Neo4j.

    Uses MERGE to avoid duplicates. All relationships have direction.
    """
    queries: list[tuple[str, dict[str, Any]]] = []

    for _, row in df.iterrows():
        source = str(row.get("source", row.get("source_id", "")))
        target = str(row.get("target", row.get("target_id", "")))
        relation = str(row.get("relation", "LOCATED_IN"))
        weight = row.get("weight", 1.0)
        desc = str(row.get("description", ""))[:500]
        generation = str(row.get("generation", "rule"))

        if not source or not target:
            continue

        # Use MERGE to create relationship only if it doesn't exist
        query = f"""
        MATCH (a {{id: $source}})
        MATCH (b {{id: $target}})
        MERGE (a)-[r:{relation}]->(b)
        SET r.weight = $weight,
            r.description = $description,
            r.generation = $generation
        """
        params = {
            "source": source,
            "target": target,
            "weight": weight,
            "description": desc,
            "generation": generation,
        }

        # Add line property for CONNECTS_TO
        if relation == "CONNECTS_TO" and "line" in row.index:
            line_val = row.get("line")
            if line_val and not (isinstance(line_val, float) and pd.isna(line_val)):
                query = query.replace(
                    "r.generation = $generation",
                    "r.generation = $generation, r.line = $line",
                )
                params["line"] = str(line_val)

        queries.append((query, params))
        queries.append((query, params))

    count = batch_execute(queries)
    logger.info("Loaded %d relationships", count)
    return count


def _load_communities(
    communities_path: Path,
    entity_community_map_path: Path,
) -> None:
    """Load community entities and CONTAINS relationships."""
    comms_df = pd.read_parquet(communities_path)

    # Create EntityCommunity nodes
    queries: list[tuple[str, dict[str, Any]]] = []
    for _, row in comms_df.iterrows():
        cid = str(row.get("community_id", ""))
        title = str(row.get("title", ""))
        summary = str(row.get("summary", ""))[:500] if "summary" in row else ""
        member_count = int(row.get("member_count", 0))

        props = {
            "id": cid,
            "cid": cid,
            "title": title,
            "summary": summary,
            "member_count": member_count,
            "level": int(row.get("level", 0)),
        }

        prop_str = ", ".join(f"n.{k} = ${k}" for k in props)
        params = {k: v for k, v in props.items()}
        query = f"MERGE (n:EntityCommunity {{id: $id}}) SET {prop_str}"
        queries.append((query, params))

    batch_execute(queries)
    logger.info("Loaded %d community nodes", len(queries))

    # Create CONTAINS relationships
    if entity_community_map_path.exists():
        map_df = pd.read_parquet(entity_community_map_path)
        queries = []
        for _, row in map_df.iterrows():
            eid = str(row.get("entity_id", ""))
            cid = str(row.get("community_id", ""))
            if eid and cid:
                queries.append((
                    """
                    MATCH (a {id: $eid})
                    MATCH (b:EntityCommunity {id: $cid})
                    MERGE (b)-[r:CONTAINS]->(a)
                    """,
                    {"eid": eid, "cid": cid},
                ))
        batch_execute(queries)
        logger.info("Loaded %d CONTAINS relationships", len(queries))


if __name__ == "__main__":
    logging.basicConfig(
        level=getattr(logging, config.settings.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    summary = load_all()
