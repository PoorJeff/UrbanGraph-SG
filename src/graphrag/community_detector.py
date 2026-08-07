"""Community Detection Agent.

Uses Leiden algorithm to detect communities in the knowledge graph.
Communities represent tightly-coupled groups of entities (e.g.,
"CBD-Transport-Weather", "Jurong-Industrial-Housing").

Outputs:
- communities.parquet: community_id, title, level, member_count
- community_map CSV: entity_id → community_id mapping

Parameters from §3.8.3 of agent.md:
- max_cluster_size: 20
- resolution: 1.0 (default)
"""

import logging
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

from src.config import config

logger = logging.getLogger(__name__)

# Community detection parameters
MAX_CLUSTER_SIZE = 20
RESOLUTION = 1.0
MIN_COMMUNITY_SIZE = 3


class CommunityDetectionAgent:
    """Detect communities in the urban knowledge graph using Leiden."""

    def detect(
        self,
        entities_path: Path | None = None,
        relationships_path: Path | None = None,
        output_dir: Path | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Run community detection on the knowledge graph.

        Args:
            entities_path: GraphRAG entities parquet
            relationships_path: GraphRAG relationships parquet
            output_dir: Output directory

        Returns:
            dict with 'communities' and 'entity_community_map' DataFrames
        """
        if entities_path is None:
            entities_path = config.data_dir / "graphrag" / "input" / "entities.parquet"
        if relationships_path is None:
            relationships_path = config.data_dir / "graphrag" / "input" / "relationships.parquet"
        if output_dir is None:
            output_dir = config.data_dir / "graphrag" / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        if not entities_path.exists() or not relationships_path.exists():
            logger.error("Missing input files for community detection")
            return {"communities": pd.DataFrame(), "entity_community_map": pd.DataFrame()}

        entities_df = pd.read_parquet(entities_path)
        rels_df = pd.read_parquet(relationships_path)

        logger.info(
            "Building graph: %d entities, %d relationships",
            len(entities_df), len(rels_df),
        )

        # Build NetworkX graph
        G = nx.Graph()

        # Add nodes
        for _, row in entities_df.iterrows():
            G.add_node(
                row["id"],
                name=row.get("name", ""),
                type=row.get("type", ""),
            )

        # Add edges
        for _, row in rels_df.iterrows():
            src = row.get("source", row.get("source_id", ""))
            tgt = row.get("target", row.get("target_id", ""))
            if src and tgt and src in G and tgt in G:
                G.add_edge(
                    src, tgt,
                    relation=row.get("relation", ""),
                    weight=row.get("weight", 1.0),
                )

        logger.info("Graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())

        # Run community detection
        communities = self._detect_communities(G)
        logger.info("Detected %d communities", len(communities))

        # Assign community IDs to entities
        entity_community_map = self._build_entity_community_map(G, communities)

        # Build community DataFrame
        community_df = self._build_community_df(communities, G)

        # Save
        comm_path = output_dir / "communities.parquet"
        community_df.to_parquet(comm_path, index=False)
        logger.info("Communities saved to %s", comm_path)

        map_path = output_dir / "entity_community_map.parquet"
        entity_community_map.to_parquet(map_path, index=False)
        logger.info("Entity-community map saved to %s", map_path)

        return {
            "communities": community_df,
            "entity_community_map": entity_community_map,
        }

    def _detect_communities(self, G: nx.Graph) -> dict[int, set[Any]]:
        """Detect communities using Louvain algorithm via NetworkX."""
        from networkx.algorithms.community import louvain_communities

        raw_communities = louvain_communities(
            G, resolution=RESOLUTION, seed=42,
        )

        communities = {}
        for i, comm in enumerate(raw_communities):
            communities[i] = set(comm)

        logger.info("Louvain: %d communities detected", len(communities))
        return communities

    def _build_entity_community_map(
        self, G: nx.Graph, communities: dict[int, set[Any]],
    ) -> pd.DataFrame:
        """Build entity_id → community_id mapping."""
        records = []
        for comm_id, nodes in communities.items():
            for node in nodes:
                node_type = G.nodes[node].get("type", "")
                node_name = G.nodes[node].get("name", "")
                records.append({
                    "entity_id": node,
                    "community_id": f"community-{comm_id}",
                    "entity_type": node_type,
                    "entity_name": node_name,
                })
        return pd.DataFrame(records)

    def _build_community_df(
        self, communities: dict[int, set[Any]], G: nx.Graph,
    ) -> pd.DataFrame:
        """Build community summary DataFrame."""
        records = []
        for comm_id, nodes in communities.items():
            # Get entity types in this community
            types = {}
            for n in nodes:
                t = G.nodes[n].get("type", "unknown")
                types[t] = types.get(t, 0) + 1

            # Generate a title from the dominant types
            sorted_types = sorted(types.items(), key=lambda x: x[1], reverse=True)
            title = " + ".join(t.replace("_", " ").title() for t, _ in sorted_types[:3])

            records.append({
                "community_id": f"community-{comm_id}",
                "title": title,
                "level": 0,
                "member_count": len(nodes),
                "type_distribution": str(types),
                "representative_entities": str(list(nodes)[:5]),
            })

        return pd.DataFrame(records)


if __name__ == "__main__":
    logging.basicConfig(
        level=getattr(logging, config.settings.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    agent = CommunityDetectionAgent()
    result = agent.detect()
    for name, df in result.items():
        print(f"\n{name}: {len(df)} rows")
        if not df.empty:
            print(df.head())
