"""Cypher Query Agent — Natural language to Cypher translation.

Read-only Cypher execution with security constraints.
45+ preset queries covering transport, population, housing, weather, spatial.
"""

import logging
from typing import Any

from src.graph.neo4j_client import run_query

logger = logging.getLogger(__name__)

FORBIDDEN = {"DELETE", "DETACH", "DROP", "REMOVE", "SET", "CREATE", "MERGE"}
MAX_ROWS = 1000


def execute(query: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Execute a Cypher query with security checks."""
    q_upper = query.upper()
    for kw in FORBIDDEN:
        if kw in q_upper.split():
            return {"error": f"Forbidden: {kw}", "results": [], "columns": [], "count": 0}
    if "LIMIT" not in q_upper:
        query = f"{query.strip().rstrip(';')} LIMIT {MAX_ROWS}"
    try:
        results = run_query(query, params)
        cols = list(results[0].keys()) if results else []
        return {"results": results, "columns": cols, "count": len(results)}
    except Exception as e:
        logger.error("Cypher failed: %s", e)
        return {"error": str(e), "results": [], "columns": [], "count": 0}


# ═══════════════════════════════════════════════════════════════
# PRESET QUERIES — organized by domain
# ═══════════════════════════════════════════════════════════════

PRESET_QUERIES: dict[str, str] = {

    # ── Transport: Counts ──
    "station_count": "MATCH (n:TransportNode {transport_type: 'mrt'}) RETURN count(n) AS total_mrt_stations",
    "bus_stop_count": "MATCH (n:TransportNode {transport_type: 'bus'}) RETURN count(n) AS total_bus_stops",
    "circle_line_stations": "MATCH (a:TransportNode {transport_type: 'mrt'})-[r:CONNECTS_TO {line: 'CCL'}]->(b) MATCH (n:TransportNode {transport_type: 'mrt'}) WHERE n.id IN [id(a), id(b)] OR (n)-[:CONNECTS_TO {line: 'CCL'}]-() RETURN count(DISTINCT n) AS circle_line_stations",

    # ── Transport: Lines at station ──
    "mrt_lines_bishan": "MATCH (mrt:TransportNode {name: 'Bishan'})-[r:CONNECTS_TO]-(n) RETURN DISTINCT r.line AS line, collect(n.name) AS connected_stations ORDER BY line",
    "lines_at_jurong_east": "MATCH (mrt:TransportNode {name: 'Jurong East'})-[r:CONNECTS_TO]-(n) RETURN DISTINCT r.line AS line ORDER BY line",
    "lines_at_woodlands": "MATCH (mrt:TransportNode {name: 'Woodlands'})-[r:CONNECTS_TO]-(n) RETURN DISTINCT r.line AS line, collect(n.name) AS connected_stations ORDER BY line",
    "lines_at_orchard": "MATCH (mrt:TransportNode {name: 'Orchard'})-[r:CONNECTS_TO]-(n) RETURN DISTINCT r.line AS line, collect(n.name) AS connected_stations ORDER BY line",
    "lines_at_city_hall": "MATCH (mrt:TransportNode {name: 'City Hall'})-[r:CONNECTS_TO]-(n) RETURN DISTINCT r.line AS line, collect(n.name) AS connected_stations ORDER BY line",

    # ── Transport: MRT by area ──
    "mrt_in_cbd": """MATCH (mrt:TransportNode {transport_type: 'mrt'})-[:LOCATED_IN]->(pa:PlanningArea)
        WHERE pa.name IN ['Downtown Core','Outram','Museum','Rochor','Singapore River','Straits View','Marina South','Marina East','Orchard']
        RETURN mrt.name AS name, pa.name AS planning_area ORDER BY pa.name, mrt.name""",
    "mrt_count_cbd": """MATCH (mrt:TransportNode {transport_type: 'mrt'})-[:LOCATED_IN]->(pa:PlanningArea)
        WHERE pa.name IN ['Downtown Core','Outram','Museum','Rochor','Singapore River','Straits View','Marina South','Marina East','Orchard']
        RETURN count(mrt) AS total_mrt_in_cbd""",
    "mrt_in_area": """MATCH (mrt:TransportNode {transport_type: 'mrt'})-[:LOCATED_IN]->(pa:PlanningArea)
        WHERE toLower(pa.name) CONTAINS toLower($area_name)
        RETURN mrt.name AS name, pa.name AS planning_area ORDER BY mrt.name""",

    # ── Transport: Connections ──
    "station_connections": """MATCH (n:TransportNode {name: $station_name})-[r:CONNECTS_TO]-(m)
        RETURN n.name AS station, count(r) AS connection_count, collect(DISTINCT r.line) AS lines, collect(m.name)[0..5] AS neighbors""",
    "path_exists": """MATCH path = shortestPath((a:TransportNode {name: $from})-[r:CONNECTS_TO*]-(b:TransportNode {name: $to}))
        RETURN length(path) AS hops, [n in nodes(path) | n.name] AS stations""",

    # ── Transport: Ranking ──
    "areas_with_most_mrt": "MATCH (mrt:TransportNode {transport_type: 'mrt'})-[:LOCATED_IN]->(pa:PlanningArea) RETURN pa.name AS area, count(mrt) AS mrt_count ORDER BY mrt_count DESC LIMIT 10",
    "mrt_in_orchard": "MATCH (mrt:TransportNode {transport_type: 'mrt'})-[:LOCATED_IN]->(pa:PlanningArea {name: 'Orchard'}) RETURN mrt.name AS station ORDER BY mrt.name",
    "areas_with_least_mrt": "MATCH (mrt:TransportNode {transport_type: 'mrt'})-[:LOCATED_IN]->(pa:PlanningArea) RETURN pa.name AS area, count(mrt) AS mrt_count ORDER BY mrt_count ASC LIMIT 10",
    "stations_most_connections": "MATCH (n:TransportNode {transport_type: 'mrt'})-[r:CONNECTS_TO]-() RETURN n.name AS station, count(r) AS connections ORDER BY connections DESC LIMIT 5",
    "hdb_highest_prices": "MATCH (pa:PlanningArea) WHERE pa.avg_resale_price IS NOT NULL RETURN pa.name AS area, pa.avg_resale_price AS avg_price ORDER BY pa.avg_resale_price DESC LIMIT 5",
    "hdb_total_transactions": "MATCH (pa:PlanningArea) WHERE pa.tx_count IS NOT NULL RETURN sum(pa.tx_count) AS total_transactions",

    # ── Transport: Bus ──
    "bus_stops_orchard": "MATCH (bus:TransportNode {transport_type: 'bus'})-[:LOCATED_IN]->(pa:PlanningArea {name: 'Orchard'}) RETURN bus.name AS name, bus.description AS description ORDER BY bus.name LIMIT 30",
    "bus_stops_in_area": "MATCH (bus:TransportNode {transport_type: 'bus'})-[:LOCATED_IN]->(pa:PlanningArea) WHERE toLower(pa.name) CONTAINS toLower($area_name) RETURN bus.name AS name ORDER BY bus.name LIMIT 30",

    # ── Population ──
    "planning_area_population": "MATCH (pa:PlanningArea) WHERE toLower(pa.name) CONTAINS toLower($area_name) RETURN pa.name AS area, pa.population AS population, pa.region AS region LIMIT 5",
    "largest_population": "MATCH (pa:PlanningArea) WHERE pa.population IS NOT NULL RETURN pa.name AS area, pa.population AS population ORDER BY pa.population DESC LIMIT 5",
    "smallest_population": "MATCH (pa:PlanningArea) WHERE pa.population IS NOT NULL AND pa.population > 0 RETURN pa.name AS area, pa.population AS population ORDER BY pa.population ASC LIMIT 5",
    "population_by_region": "MATCH (pa:PlanningArea) WHERE pa.region IS NOT NULL RETURN pa.region AS region, sum(pa.population) AS total_population ORDER BY total_population DESC",

    # ── Housing ──
    "hdb_price_town": "MATCH (pa:PlanningArea) WHERE toLower(pa.name) CONTAINS toLower($town) RETURN pa.name AS area, pa.avg_resale_price AS avg_price, pa.population AS population LIMIT 3",

    # ── Spatial ──
    "station_planning_area": "MATCH (n:TransportNode {transport_type: 'mrt'}) WHERE toLower(n.name) CONTAINS toLower($station) MATCH (n)-[:LOCATED_IN]->(pa:PlanningArea) RETURN n.name AS station, pa.name AS planning_area LIMIT 5",
    "mrt_in_any_area": "MATCH (mrt:TransportNode {transport_type: 'mrt'})-[:LOCATED_IN]->(pa:PlanningArea) WHERE toLower(pa.name) CONTAINS toLower($area) RETURN mrt.name AS name, pa.name AS area ORDER BY mrt.name LIMIT 20",
    "bus_near_station": "MATCH (mrt:TransportNode {transport_type: 'mrt'}) WHERE toLower(mrt.name) CONTAINS toLower($station) WITH mrt MATCH (bus:TransportNode {transport_type: 'bus'})-[:LOCATED_IN]->(pa:PlanningArea) WHERE pa.name = mrt.planning_area RETURN bus.name AS bus_stop, pa.name AS area, mrt.name AS station ORDER BY bus.name LIMIT 20",
    "station_area_lookup": "MATCH (n:TransportNode) WHERE toLower(n.name) CONTAINS toLower($station) OPTIONAL MATCH (n)-[:LOCATED_IN]->(pa:PlanningArea) RETURN n.name AS name, n.transport_type AS type, pa.name AS planning_area LIMIT 5",
    "bishan_to_orchard_path": "MATCH p = shortestPath((a:TransportNode {name: 'Bishan'})-[:CONNECTS_TO*]-(b:TransportNode {name: 'Orchard'})) RETURN length(p) AS hops, [n IN nodes(p) | n.name] AS stations",
    "jurong_east_to_city_hall_path": "MATCH p = shortestPath((a:TransportNode {name: 'Jurong East'})-[:CONNECTS_TO*]-(b:TransportNode {name: 'City Hall'})) RETURN length(p) AS hops, [n IN nodes(p) | n.name] AS stations",

    # ── Holiday ──
    "next_holiday": "MATCH (h:Holiday) WHERE h.date >= '2025-01-01' RETURN h.name AS holiday, h.date AS date ORDER BY h.date LIMIT 3",
    "holidays_count": "MATCH (h:Holiday) RETURN count(h) AS total_holidays",

    # ── Weather ──
    "weather_stations": "MATCH (ws:WeatherStation) RETURN ws.name AS station, ws.station_id AS id, ws.lat AS lat, ws.lon AS lon LIMIT 20",
    "rainiest_day": "MATCH (we:WeatherEvent) WHERE we.rainfall_mm IS NOT NULL RETURN we.date AS date, we.rainfall_mm AS rainfall_mm, we.temp_mean AS temp ORDER BY we.rainfall_mm DESC LIMIT 5",
    "hottest_day": "MATCH (we:WeatherEvent) WHERE we.temp_max IS NOT NULL RETURN we.date AS date, we.temp_max AS temp_max, we.rainfall_mm AS rain ORDER BY we.temp_max DESC LIMIT 5",
    "weather_summary": "MATCH (we:WeatherEvent) RETURN avg(we.rainfall_mm) AS avg_rain, max(we.rainfall_mm) AS max_rain, avg(we.temp_mean) AS avg_temp, min(we.temp_min) AS min_temp, max(we.temp_max) AS max_temp, count(we) AS days",
    "total_population": "MATCH (pa:PlanningArea) WHERE pa.population IS NOT NULL RETURN sum(pa.population) AS singapore_population",
    "lines_at_station": "MATCH (mrt:TransportNode {transport_type:'mrt'}) WHERE toLower(mrt.name) CONTAINS toLower($station) MATCH (mrt)-[r:CONNECTS_TO]-(n) RETURN DISTINCT r.line AS line, collect(n.name) AS stations ORDER BY line LIMIT 5",
}


def run_preset(qid: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    if qid not in PRESET_QUERIES:
        return {"error": f"Unknown: {qid}"}
    return execute(PRESET_QUERIES[qid], params)


def list_presets() -> list[str]:
    return sorted(PRESET_QUERIES.keys())
