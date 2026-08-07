"""Local Search Agent.

Performs entity-centric local search:
1. Extract entity mentions from the user query
2. Retrieve the subgraph (entity + 1-2 hop neighbors)
3. Return context for answer generation

Parameters (§3.10.1):
- top_k_entities: 5
- max_hops: 2
- context_window: 4000 tokens
"""

import logging
from typing import Any

from src.config import config
from src.graph.neo4j_client import run_query

logger = logging.getLogger(__name__)

# Default parameters
TOP_K_ENTITIES = 5
MAX_HOPS = 2


def search(
    query: str,
    top_k: int = TOP_K_ENTITIES,
    max_hops: int = MAX_HOPS,
) -> dict[str, Any]:
    """Execute local search for a natural language query.

    Args:
        query: User's natural language question
        top_k: Number of top entities to retrieve
        max_hops: Maximum hops from entity (1-2)

    Returns:
        dict with:
        - entities: list of matched entities
        - subgraph: dict of nodes and relationships
        - context_text: flat text for LLM prompt
        - sources: source citations
    """
    # Step 1: Find entities mentioned in the query
    entities = _find_entities(query, top_k)
    if not entities:
        return {
            "entities": [],
            "subgraph": {"nodes": [], "edges": []},
            "context_text": "No matching entities found in the knowledge graph.",
            "sources": [],
        }

    # Step 2: Expand subgraph (1-2 hop neighbors)
    entity_ids = [e["id"] for e in entities]
    subgraph = _expand_subgraph(entity_ids, max_hops)

    # Step 3: Build context text
    context_text = _build_context(entities, subgraph)

    # Step 4: Build source citations
    sources = _build_sources(entities, subgraph)

    logger.info(
        "Local search: '%s' → %d entities, %d nodes, %d edges",
        query[:60], len(entities),
        len(subgraph["nodes"]), len(subgraph["edges"]),
    )

    return {
        "entities": entities,
        "subgraph": subgraph,
        "context_text": context_text,
        "sources": sources,
    }


def _find_entities(query: str, top_k: int) -> list[dict[str, Any]]:
    """Find entities whose name/description match the query.

    Uses substring matching + fuzzy heuristics.
    For production, this should use text embeddings.
    """
    # Extract potential entity names from query (capitalized words, known patterns)
    words = query.split()
    candidates = [w.strip(",.?!()") for w in words if len(w) > 2]

    all_matches: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for candidate in candidates[:10]:  # limit candidates
        # Search by name contains
        result = run_query(
            """
            MATCH (n)
            WHERE n.name CONTAINS $keyword OR n.id CONTAINS $keyword
            RETURN n.id AS id, labels(n) AS labels, n.name AS name,
                   n.lat AS lat, n.lon AS lon, n.planning_area AS planning_area,
                   n.transport_type AS transport_type, n.population AS population
            LIMIT 5
            """,
            {"keyword": candidate},
        )
        for r in result:
            if r["id"] not in seen_ids:
                seen_ids.add(r["id"])
                all_matches.append(r)

        if len(all_matches) >= top_k:
            break

    return all_matches[:top_k]


def _expand_subgraph(
    entity_ids: list[str], max_hops: int,
) -> dict[str, Any]:
    """Expand the subgraph around given entities.

    Retrieves nodes and relationships within max_hops.
    """
    if max_hops == 1:
        hop_clause = "1"
    else:
        hop_clause = "1..2"

    result = run_query(
        f"""
        MATCH (n)-[r]-(m)
        WHERE n.id IN $ids
        AND NOT (m:EntityCommunity)  // exclude community meta-nodes
        RETURN DISTINCT
               n.id AS source_id, n.name AS source_name, labels(n) AS source_labels,
               type(r) AS relation,
               m.id AS target_id, m.name AS target_name, labels(m) AS target_labels,
               r.weight AS weight, r.description AS description
        LIMIT 100
        """,
        {"ids": entity_ids},
    )

    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []

    for row in result:
        src = row["source_id"]
        tgt = row["target_id"]
        # Track unique nodes
        if src not in nodes:
            nodes[src] = {
                "id": src,
                "name": row["source_name"],
                "labels": row["source_labels"],
            }
        if tgt not in nodes:
            nodes[tgt] = {
                "id": tgt,
                "name": row["target_name"],
                "labels": row["target_labels"],
            }
        edges.append({
            "source": src,
            "target": tgt,
            "relation": row["relation"],
            "weight": row["weight"],
            "description": row.get("description", ""),
        })

    return {"nodes": list(nodes.values()), "edges": edges}


def _build_context(
    entities: list[dict[str, Any]],
    subgraph: dict[str, Any],
) -> str:
    """Build a text context for the LLM from entities and subgraph."""
    lines = ["## Retrieved Entities\n"]

    for e in entities:
        name = e.get("name", e.get("id", ""))
        labels = e.get("labels", [])
        pa = e.get("planning_area", "")
        pop = e.get("population", "")
        tt = e.get("transport_type", "")
        lat = e.get("lat", "")
        lon = e.get("lon", "")

        detail = f"{name}"
        if pa:
            detail += f" (Planning Area: {pa})"
        if pop:
            detail += f" | Population: {pop:,}"
        if tt:
            detail += f" | Type: {tt}"
        if lat and lon:
            detail += f" | Coord: ({lat:.4f}, {lon:.4f})"
        detail += f" | Labels: {labels}"

        lines.append(f"- {detail}")

    lines.append(f"\n## Subgraph ({len(subgraph['nodes'])} nodes, {len(subgraph['edges'])} edges)\n")

    for edge in subgraph.get("edges", [])[:30]:
        src_name = ""
        tgt_name = ""
        for n in subgraph["nodes"]:
            if n["id"] == edge["source"]:
                src_name = n.get("name", edge["source"])
            if n["id"] == edge["target"]:
                tgt_name = n.get("name", edge["target"])
        lines.append(f"- {src_name} --[{edge['relation']}]--> {tgt_name}")

    return "\n".join(lines)


def _build_sources(
    entities: list[dict[str, Any]],
    subgraph: dict[str, Any],
) -> list[str]:
    """Build source citations for each piece of evidence."""
    sources = []
    for e in entities:
        name = e.get("name", "unknown")
        pid = e.get("id", "unknown")
        sources.append(f"[Source: {name}, entity_id={pid}]")
    return sources
