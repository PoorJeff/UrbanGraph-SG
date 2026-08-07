"""Global Search Agent.

Performs map-reduce search across communities:
1. Match query against all community summaries
2. Select top-K communities
3. Assemble context from community summaries + member entities
4. Return aggregated context

Used for comparison/summary questions (e.g., "Which areas have the most...")
"""

import logging
from typing import Any

from src.graph.neo4j_client import run_query

logger = logging.getLogger(__name__)

TOP_K_COMMUNITIES = 5


def search(
    query: str,
    top_k: int = TOP_K_COMMUNITIES,
) -> dict[str, Any]:
    """Execute global search across community summaries.

    Args:
        query: User question (typically comparison/summary type)
        top_k: Number of top communities to return

    Returns:
        dict with communities, context_text, and sources
    """
    # Match communities by summary content (substring matching)
    # Extract key terms from query
    terms = [w.strip(",.?!()") for w in query.split() if len(w) > 3]

    communities = []
    if terms:
        # Build OR-contains query
        where_clauses = " OR ".join(
            [f"c.summary CONTAINS $t{i}" for i in range(min(len(terms), 5))]
        )
        params = {f"t{i}": term for i, term in enumerate(terms[:5])}

        result = run_query(
            f"""
            MATCH (c:EntityCommunity)
            WHERE {where_clauses}
            RETURN c.id AS id, c.title AS title, c.summary AS summary,
                   c.member_count AS member_count
            ORDER BY c.member_count DESC
            LIMIT $top_k
            """,
            {**params, "top_k": top_k},
        )

        communities = list(result)

    # If no keyword match, fall back to largest communities
    if not communities:
        result = run_query(
            """
            MATCH (c:EntityCommunity)
            RETURN c.id AS id, c.title AS title, c.summary AS summary,
                   c.member_count AS member_count
            ORDER BY c.member_count DESC
            LIMIT $top_k
            """,
            {"top_k": top_k},
        )
        communities = list(result)

    # Build context from community summaries
    context_parts = []
    sources = []

    for comm in communities:
        cid = comm["id"]
        title = comm.get("title", cid)
        summary = comm.get("summary", "")
        count = comm.get("member_count", 0)

        context_parts.append(
            f"## Community: {title} ({count} members)\n{summary}\n"
        )

        # Get sample entities from this community (for richer context)
        entities = run_query(
            """
            MATCH (c:EntityCommunity {id: $cid})-[:CONTAINS]->(e)
            WHERE NOT e:EntityCommunity
            RETURN e.name AS name, labels(e) AS labels, e.planning_area AS area
            LIMIT 5
            """,
            {"cid": cid},
        )
        if entities:
            context_parts.append("Key entities: " + ", ".join(
                e.get("name", "") for e in entities
            ) + "\n")

        sources.append(f"[Source: Community '{title}', {count} members]")

    context_text = "\n".join(context_parts)

    logger.info(
        "Global search: '%s' → %d communities",
        query[:60], len(communities),
    )

    return {
        "communities": communities,
        "community_count": len(communities),
        "context_text": context_text,
        "sources": sources,
    }
