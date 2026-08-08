"""Neo4j connection management.

Supports two backends:
- Bolt (neo4j://, bolt://) — used for local Neo4j or unblocked cloud
- HTTP API (https://) — used for AuraDB when Bolt port 7687 is blocked

Auto-detects Bolt failure on AuraDB URIs and falls back to HTTP.
"""

import json
import logging
from base64 import b64encode
from typing import Any

import requests
from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import ServiceUnavailable, AuthError

from src.config import config

logger = logging.getLogger(__name__)

_DRIVER: Driver | None = None
_USE_HTTP: bool = False
_HTTP_AUTH: str = ""
_HTTP_BASE: str = ""


def _build_auth_header(user: str, password: str) -> str:
    raw = f"{user}:{password}"
    return "Basic " + b64encode(raw.encode()).decode()


def get_driver() -> Driver:
    global _DRIVER, _USE_HTTP, _HTTP_AUTH, _HTTP_BASE
    if _DRIVER is not None or _USE_HTTP:
        return _DRIVER

    settings = config.settings
    uri = settings.neo4j_uri
    user = settings.neo4j_user
    password = settings.neo4j_password

    logger.info("Connecting to Neo4j at %s (user: %s)", uri, user)

    # Try Bolt first
    try:
        _DRIVER = GraphDatabase.driver(
            uri,
            auth=(user, password) if password else None,
            max_connection_lifetime=3600,
            max_connection_pool_size=10,
            connection_acquisition_timeout=10,
        )
        _DRIVER.verify_connectivity()
        logger.info("Neo4j Bolt connection verified")
        return _DRIVER
    except (ServiceUnavailable, AuthError, OSError) as e:
        logger.warning("Bolt connection failed: %s", e)
        _DRIVER = None
        if ".databases.neo4j.io" in uri:
            logger.info("AuraDB detected, trying HTTP Query API fallback...")
        else:
            raise

    # Fallback: HTTP Query API for AuraDB
    instance_id = _extract_instance_id(uri)
    if not instance_id:
        raise RuntimeError(f"Cannot determine AuraDB instance ID from URI: {uri}")

    _HTTP_AUTH = _build_auth_header(user, password)
    _HTTP_BASE = f"https://{instance_id}.databases.neo4j.io/db/{instance_id}/query/v2"
    _USE_HTTP = True

    # Verify HTTP API works
    try:
        resp = requests.post(
            _HTTP_BASE,
            headers={
                "Content-Type": "application/json",
                "Authorization": _HTTP_AUTH,
            },
            json={"statement": "RETURN 1 AS ok"},
            timeout=15,
        )
        resp.raise_for_status()
        logger.info("Neo4j HTTP API connection verified")
    except Exception as e:
        logger.error("HTTP API also failed: %s", e)
        _USE_HTTP = False
        raise RuntimeError(f"Neither Bolt nor HTTP API can connect: {e}")

    return _DRIVER


def _extract_instance_id(uri: str) -> str | None:
    import re
    m = re.search(r'([a-f0-9]{8})\.databases\.neo4j\.io', uri)
    return m.group(1) if m else None


def get_session() -> Session:
    return get_driver().session()


def close_driver() -> None:
    global _DRIVER, _USE_HTTP
    if _DRIVER:
        _DRIVER.close()
        _DRIVER = None
    _USE_HTTP = False
    logger.info("Neo4j driver closed")


def run_query(
    query: str,
    params: dict[str, Any] | None = None,
    database: str = "neo4j",
) -> list[dict[str, Any]]:
    params = params or {}

    # Ensure driver is initialized (will set _USE_HTTP if Bolt fails)
    try:
        get_driver()
    except Exception:
        pass

    if _USE_HTTP:
        return _run_http_query(query, params)

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


def _run_http_query(query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Execute Cypher via HTTP Query API (AuraDB fallback)."""
    body: dict[str, Any] = {"statement": query}
    if params:
        body["parameters"] = params

    try:
        resp = requests.post(
            _HTTP_BASE,
            headers={
                "Content-Type": "application/json",
                "Authorization": _HTTP_AUTH,
            },
            json=body,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        records: list[dict[str, Any]] = []
        if "data" in data and "fields" in data:
            fields = data["fields"]
            for row in data["data"]["values"]:
                record = {}
                for i, field in enumerate(fields):
                    record[field] = row[i]
                records.append(record)
        return records
    except Exception as e:
        logger.error("HTTP query failed: %s", e)
        raise


def execute_write(
    query: str,
    params: dict[str, Any] | None = None,
    database: str = "neo4j",
) -> None:
    params = params or {}
    get_driver()
    if _USE_HTTP:
        _run_http_query(query, params)
        return
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
    get_driver()

    if _USE_HTTP:
        total = 0
        for query, params in queries:
            try:
                _run_http_query(query, params)
                total += 1
            except Exception as e:
                logger.error("Batch HTTP failed at %d: %s", total, e)
                raise
        logger.info("HTTP batch complete: %d queries", total)
        return total

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
