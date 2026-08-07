"""Neo4j connection management.

Provides a singleton driver with connection pooling.
Handles connection verification, session management, and graceful shutdown.
"""

import logging
from typing import Any, Generator

from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import ServiceUnavailable, AuthError

from src.config import config

logger = logging.getLogger(__name__)

_DRIVER: Driver | None = None


def get_driver() -> Driver:
    """Get or create the Neo4j driver singleton."""
    global _DRIVER
    if _DRIVER is None:
        settings = config.settings
        uri = settings.neo4j_uri
        user = settings.neo4j_user
        password = settings.neo4j_password

        logger.info("Connecting to Neo4j at %s (user: %s)", uri, user)

        try:
            _DRIVER = GraphDatabase.driver(
                uri,
                auth=(user, password) if password else None,
                max_connection_lifetime=3600,
                max_connection_pool_size=10,
                connection_acquisition_timeout=30,
            )
            # Verify connectivity
            _DRIVER.verify_connectivity()
            logger.info("Neo4j connection verified")
        except ServiceUnavailable as e:
            logger.error("Neo4j is not running at %s: %s", uri, e)
            _DRIVER = None
            raise
        except AuthError as e:
            logger.error("Neo4j authentication failed: %s", e)
            _DRIVER = None
            raise

    return _DRIVER


def get_session() -> Session:
    """Get a new Neo4j session."""
    return get_driver().session()


def close_driver() -> None:
    """Close the Neo4j driver."""
    global _DRIVER
    if _DRIVER:
        _DRIVER.close()
        _DRIVER = None
        logger.info("Neo4j driver closed")


def run_query(
    query: str,
    params: dict[str, Any] | None = None,
    database: str = "neo4j",
) -> list[dict[str, Any]]:
    """Execute a Cypher query and return results as list of dicts.

    Args:
        query: Cypher query string
        params: Optional query parameters
        database: Database name (default: neo4j)

    Returns:
        List of result records as dictionaries
    """
    params = params or {}
    records: list[dict[str, Any]] = []

    with get_session() as session:
        try:
            result = session.run(query, params, database=database)
            for record in result:
                records.append(dict(record))
        except Exception as e:
            logger.error("Query failed: %s", e)
            logger.debug("Query: %s\nParams: %s", query[:200], params)
            raise

    return records


def execute_write(
    query: str,
    params: dict[str, Any] | None = None,
    database: str = "neo4j",
) -> None:
    """Execute a write query (no return value expected)."""
    params = params or {}
    with get_session() as session:
        try:
            session.run(query, params, database=database)
        except Exception as e:
            logger.error("Write query failed: %s", e)
            raise


def batch_execute(
    queries: list[tuple[str, dict[str, Any]]],
    database: str = "neo4j",
    batch_size: int = 1000,
) -> int:
    """Execute multiple write queries in batches.

    Args:
        queries: List of (query, params) tuples
        database: Database name
        batch_size: Number of queries to execute per transaction

    Returns:
        Total number of queries executed
    """
    total = 0
    with get_session() as session:
        for i in range(0, len(queries), batch_size):
            batch = queries[i : i + batch_size]
            try:
                with session.begin_transaction() as tx:
                    for query, params in batch:
                        tx.run(query, params)
                    tx.commit()
                total += len(batch)
                if (i + batch_size) % 5000 == 0:
                    logger.info("Batch progress: %d/%d executed", total, len(queries))
            except Exception as e:
                logger.error("Batch failed at offset %d: %s", i, e)
                raise

    logger.info("Batch execution complete: %d queries", total)
    return total
