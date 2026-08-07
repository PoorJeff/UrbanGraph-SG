"""Neo4j graph schema definition.

Defines:
- Node labels: TransportNode, WeatherStation, WeatherEvent, PlanningArea,
  POI, HDBTown, Holiday, EntityCommunity
- Relationship types: CONNECTS_TO, LOCATED_IN, NEAR, PART_OF, RECORDS,
  AFFECTS, CORRELATES_WITH, CONTAINS
- Constraints and indexes for query performance
"""

import logging

from src.graph.neo4j_client import execute_write, run_query

logger = logging.getLogger(__name__)


def create_constraints_and_indexes() -> None:
    """Create all schema constraints and indexes.

    Must be called after Neo4j is running and before data loading.
    Idempotent — safe to call multiple times.
    """
    logger.info("Creating Neo4j schema constraints and indexes...")

    # Uniqueness constraints
    constraints = [
        "CREATE CONSTRAINT transportnode_id IF NOT EXISTS FOR (n:TransportNode) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT weatherstation_id IF NOT EXISTS FOR (n:WeatherStation) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT weatherevent_id IF NOT EXISTS FOR (n:WeatherEvent) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT planningarea_id IF NOT EXISTS FOR (n:PlanningArea) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT poi_id IF NOT EXISTS FOR (n:POI) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT hdbtown_id IF NOT EXISTS FOR (n:HDBTown) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT holiday_id IF NOT EXISTS FOR (n:Holiday) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT entitycommunity_id IF NOT EXISTS FOR (n:EntityCommunity) REQUIRE n.id IS UNIQUE",
    ]

    for cypher in constraints:
        try:
            execute_write(cypher)
            logger.debug("Constraint: %s", cypher[:80])
        except Exception as e:
            if "already exists" in str(e).lower() or "equivalent" in str(e).lower():
                logger.debug("Constraint already exists: %s", cypher[:80])
            else:
                logger.warning("Constraint error: %s — %s", cypher[:80], e)

    # Indexes for frequent query patterns
    indexes = [
        "CREATE INDEX transportnode_name IF NOT EXISTS FOR (n:TransportNode) ON (n.name)",
        "CREATE INDEX transportnode_type IF NOT EXISTS FOR (n:TransportNode) ON (n.type)",
        "CREATE INDEX planningarea_name IF NOT EXISTS FOR (n:PlanningArea) ON (n.name)",
        "CREATE INDEX weatherevent_date IF NOT EXISTS FOR (n:WeatherEvent) ON (n.date)",
        "CREATE INDEX hdbtown_name IF NOT EXISTS FOR (n:HDBTown) ON (n.name)",
    ]

    for cypher in indexes:
        try:
            execute_write(cypher)
            logger.debug("Index: %s", cypher[:80])
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.debug("Index already exists: %s", cypher[:80])
            else:
                logger.warning("Index error: %s — %s", cypher[:80], e)

    logger.info("Schema setup complete")


def drop_all() -> None:
    """⚠️ Drop all nodes, relationships, constraints and indexes."""
    logger.warning("Dropping all Neo4j data!")
    execute_write("MATCH (n) DETACH DELETE n")
    for cypher in [
        "SHOW CONSTRAINTS",
        "SHOW INDEXES",
    ]:
        results = run_query(cypher)
        # Drop each constraint/index by name
        for r in results:
            name = r.get("name", "")
            if name:
                try:
                    execute_write(f"DROP CONSTRAINT {name} IF EXISTS")
                except Exception:
                    try:
                        execute_write(f"DROP INDEX {name} IF EXISTS")
                    except Exception:
                        pass
    logger.warning("All data dropped")


def get_schema_summary() -> dict[str, int]:
    """Return a summary of the current graph."""
    counts = {}
    for label in [
        "TransportNode", "WeatherStation", "WeatherEvent",
        "PlanningArea", "POI", "HDBTown", "Holiday", "EntityCommunity",
    ]:
        result = run_query(f"MATCH (n:{label}) RETURN count(n) AS c")
        counts[label] = result[0]["c"] if result else 0

    result = run_query("MATCH ()-[r]->() RETURN count(r) AS c")
    counts["relationships"] = result[0]["c"] if result else 0

    return counts
