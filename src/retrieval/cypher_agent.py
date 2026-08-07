"""Cypher Query Agent.

Translates natural language queries into Cypher for direct graph database access.

Security constraints (§3.10.3):
- Read-only: blocks DELETE, DETACH DELETE, DROP, REMOVE, SET, CREATE, MERGE
- Max rows: 1000
- Query timeout: 5 seconds

Supported query patterns:
- Entity listing by type/location
- Path queries between stations
- Aggregation queries (count, avg, etc.)
- Filtered queries by property values
"""

import logging
from typing import Any

from src.graph.neo4j_client import run_query

logger = logging.getLogger(__name__)

# Forbidden keywords (security)
FORBIDDEN = {"DELETE", "DETACH", "DROP", "REMOVE", "SET", "CREATE", "MERGE"}
MAX_ROWS = 1000
QUERY_TIMEOUT = 5000  # ms


def execute(query: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Execute a Cypher query with security checks.

    Args:
        query: Cypher query string
        params: Query parameters

    Returns:
        dict with results, columns, and row count
    """
    # Security check
    query_upper = query.upper()
    for forbidden in FORBIDDEN:
        if forbidden in query_upper.split():
            return {
                "error": f"Forbidden keyword in query: {forbidden}",
                "results": [],
                "columns": [],
                "count": 0,
            }

    # Add limit if not present
    if "LIMIT" not in query_upper:
        query = query.rstrip().rstrip(";")
        query = f"{query} LIMIT {MAX_ROWS}"

    try:
        results = run_query(query, params)
        columns = list(results[0].keys()) if results else []
        return {
            "results": results,
            "columns": columns,
            "count": len(results),
        }
    except Exception as e:
        logger.error("Cypher query failed: %s", e)
        return {
            "error": str(e),
            "results": [],
            "columns": [],
            "count": 0,
        }


# Pre-built query templates for preset questions
PRESET_QUERIES: dict[str, str] = {
    "mrt_in_cbd": """
        MATCH (mrt:TransportNode {transport_type: 'mrt'})-[:LOCATED_IN]->(pa:PlanningArea)
        WHERE pa.name CONTAINS 'Downtown' OR pa.name CONTAINS 'Outram'
           OR pa.name CONTAINS 'Museum' OR pa.name CONTAINS 'Rochor'
           OR pa.name CONTAINS 'Singapore River' OR pa.name CONTAINS 'Straits View'
           OR pa.name CONTAINS 'Marina' OR pa.name CONTAINS 'Orchard'
        RETURN mrt.name AS name, pa.name AS planning_area, mrt.lat AS lat, mrt.lon AS lon
        ORDER BY pa.name, mrt.name
    """,

    "mrt_lines_bishan": """
        MATCH (mrt:TransportNode {name: 'Bishan'})-[r:CONNECTS_TO]-(neighbor:TransportNode)
        RETURN DISTINCT r.line AS line, collect(neighbor.name) AS connected_stations
        ORDER BY line
    """,

    "station_count": """
        MATCH (n:TransportNode {transport_type: 'mrt'})
        RETURN count(n) AS total_mrt_stations
    """,

    "poi_near_jurong_east": """
        MATCH (mrt:TransportNode {name: 'Jurong East'})
        MATCH (poi:TransportNode {transport_type: 'bus'})-[:LOCATED_IN]->(pa:PlanningArea)
        WHERE pa.name = 'Jurong East'
        RETURN poi.name AS bus_stop, poi.description AS description
        LIMIT 20
    """,

    "areas_with_most_mrt": """
        MATCH (mrt:TransportNode {transport_type: 'mrt'})-[:LOCATED_IN]->(pa:PlanningArea)
        RETURN pa.name AS planning_area, count(mrt) AS mrt_count
        ORDER BY mrt_count DESC
        LIMIT 10
    """,

    "hdb_price_punggol": """
        MATCH (pa:PlanningArea {name: 'Punggol'})
        RETURN pa.name AS area, pa.population AS population
    """,

    "bus_stops_orchard": """
        MATCH (bus:TransportNode {transport_type: 'bus'})
        WHERE bus.road_name IS NOT NULL AND bus.road_name CONTAINS 'Orchard'
        RETURN bus.name AS name, bus.road_name AS road, bus.description AS description
        ORDER BY bus.name
        LIMIT 30
    """,

    "planning_area_population": """
        MATCH (pa:PlanningArea)
        WHERE toLower(pa.name) CONTAINS toLower($area_name)
        RETURN pa.name AS area, pa.population AS population, pa.region AS region
        LIMIT 5
    """,

    "lines_at_jurong_east": """
        MATCH (mrt:TransportNode {name: 'Jurong East'})-[r:CONNECTS_TO]-(neighbor)
        RETURN DISTINCT r.line AS line
        ORDER BY line
    """,
}


def run_preset(query_id: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Execute a preset Cypher query by ID."""
    if query_id not in PRESET_QUERIES:
        return {"error": f"Unknown preset query: {query_id}"}
    return execute(PRESET_QUERIES[query_id], params)


def list_presets() -> list[str]:
    """List available preset query IDs."""
    return sorted(PRESET_QUERIES.keys())
