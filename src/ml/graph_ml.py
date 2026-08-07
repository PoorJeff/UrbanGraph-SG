"""Graph Machine Learning — Node2Vec Embeddings + Link Prediction.

Applies Node2Vec to the Singapore transport network to:
1. Generate low-dimensional embeddings for MRT/bus stations
2. Predict missing connections (link prediction)
3. Visualize station embeddings via t-SNE

Demonstrates: graph representation learning, feature extraction from graphs,
unsupervised embedding, dimensionality reduction.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import networkx as nx

logger = logging.getLogger(__name__)
REPORTS_DIR = Path(__file__).parent.parent.parent / "reports" / "figures"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


class GraphMLEngine:
    """Node2Vec + Link Prediction on the transport network graph."""

    def __init__(self):
        self.G: nx.Graph | None = None
        self.embeddings: dict[str, np.ndarray] = {}
        self.node_list: list[str] = []

    def build_graph(self) -> nx.Graph:
        """Build a NetworkX graph from Neo4j transport data."""
        try:
            from src.graph.neo4j_client import run_query
        except Exception:
            return nx.Graph()

        G = nx.Graph()

        # Add MRT stations as nodes
        try:
            nodes = run_query("""MATCH (n:TransportNode)
                WHERE n.transport_type IN ['mrt','bus'] AND n.lat IS NOT NULL
                RETURN n.id AS id, n.name AS name, n.transport_type AS type, n.lat AS lat, n.lon AS lon
                LIMIT 3000""")
        except Exception:
            return G

        for n in nodes:
            G.add_node(n["id"], name=n.get("name",""), type=n.get("type",""),
                       lat=n.get("lat"), lon=n.get("lon"))

        # Add CONNECTS_TO edges
        try:
            edges = run_query("""MATCH (a:TransportNode)-[r:CONNECTS_TO]->(b:TransportNode)
                RETURN a.id AS a, b.id AS b, r.line AS line LIMIT 500""")
        except Exception:
            edges = []

        for e in edges:
            G.add_edge(e["a"], e["b"], line=e.get("line",""))

        self.G = G
        self.node_list = list(G.nodes())
        logger.info("Graph built: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
        return G

    def train_node2vec(self, dimensions=64, walk_length=20, num_walks=50, p=1.0, q=1.0):
        """Train Node2Vec embeddings on the transport graph."""
        if self.G is None:
            self.build_graph()

        if self.G.number_of_nodes() < 10:
            logger.warning("Graph too small for Node2Vec")
            return {}

        try:
            from node2vec import Node2Vec
        except ImportError:
            logger.warning("node2vec not installed. Using fallback.")
            return self._fallback_embeddings(dimensions)

        # Train Node2Vec
        n2v = Node2Vec(
            self.G, dimensions=dimensions, walk_length=walk_length,
            num_walks=num_walks, p=p, q=q, workers=2,
        )
        model = n2v.fit(window=10, min_count=1, batch_words=4)

        self.embeddings = {node: model.wv[node] for node in self.node_list if node in model.wv}
        logger.info("Node2Vec trained: %d embeddings, dim=%d", len(self.embeddings), dimensions)
        return self.embeddings

    def _fallback_embeddings(self, dimensions=32) -> dict[str, np.ndarray]:
        """Fallback: simple adjacency-based embedding using SVD."""
        if self.G is None: return {}
        try:
            from sklearn.decomposition import TruncatedSVD
            from scipy.sparse import csr_matrix
        except ImportError:
            return {}

        n = self.G.number_of_nodes()
        nodes = self.node_list
        idx = {node: i for i, node in enumerate(nodes)}

        # Build adjacency matrix
        rows, cols = [], []
        for u, v in self.G.edges():
            if u in idx and v in idx:
                rows.extend([idx[u], idx[v]])
                cols.extend([idx[v], idx[u]])

        A = csr_matrix(([1]*len(rows), (rows, cols)), shape=(n, n))
        svd = TruncatedSVD(n_components=min(dimensions, n-1), random_state=42)
        emb = svd.fit_transform(A)

        self.embeddings = {nodes[i]: emb[i] for i in range(n)}
        logger.info("SVD fallback: %d embeddings, dim=%d", len(self.embeddings), dimensions)
        return self.embeddings

    def predict_links(self, top_k=20) -> list[dict]:
        """Predict missing links using embedding similarity."""
        if not self.embeddings:
            self.train_node2vec()
        if not self.embeddings or self.G is None:
            return []

        # Find unconnected node pairs with high embedding similarity
        existing_edges = set()
        for u, v in self.G.edges():
            existing_edges.add((u, v))
            existing_edges.add((v, u))

        candidates = []
        emb_list = list(self.embeddings.items())
        for i in range(len(emb_list)):
            for j in range(i+1, len(emb_list)):
                n1, e1 = emb_list[i]
                n2, e2 = emb_list[j]
                if (n1, n2) in existing_edges:
                    continue
                sim = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-10))
                if sim > 0.7:
                    name1 = self.G.nodes[n1].get("name", n1)
                    name2 = self.G.nodes[n2].get("name", n2)
                    candidates.append((sim, n1, n2, name1, name2))

        candidates.sort(key=lambda x: -x[0])
        top = candidates[:top_k]
        for sim, n1, n2, name1, name2 in top:
            logger.info("  Link prediction: %s ↔ %s (%.3f)", name1[:30], name2[:30], sim)

        return [{"source": n1, "target": n2, "source_name": name1, "target_name": name2, "similarity": round(sim,3)}
                for sim, n1, n2, name1, name2 in top]

    def plot_embeddings(self):
        """Visualize embeddings with t-SNE."""
        if not self.embeddings:
            return

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE

        nodes = list(self.embeddings.keys())
        X = np.array([self.embeddings[n] for n in nodes])

        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(nodes)-1))
        X_2d = tsne.fit_transform(X)

        # Color by node type
        types = [self.G.nodes[n].get("type", "?") for n in nodes]
        colors = {"mrt": "#ED2939", "bus": "#005EC4"}
        point_colors = [colors.get(t, "#888") for t in types]

        fig, ax = plt.subplots(figsize=(12, 10))
        ax.scatter(X_2d[:,0], X_2d[:,1], c=point_colors, s=2, alpha=0.6)

        # Label a few notable stations
        for i, n in enumerate(nodes[:50]):
            name = self.G.nodes[n].get("name", "")[:12]
            if self.G.degree(n) > 3:
                ax.annotate(name, (X_2d[i,0], X_2d[i,1]), fontsize=5, alpha=0.7)

        ax.set_title("Transport Network Node Embeddings (t-SNE)", fontsize=12, fontweight="bold")
        from matplotlib.lines import Line2D
        legend = [Line2D([0],[0],marker="o",color="w",markerfacecolor=c,markersize=8,label=l)
                  for l,c in [("MRT","#ED2939"),("Bus","#005EC4")]]
        ax.legend(handles=legend)

        plt.tight_layout()
        path = REPORTS_DIR / "node_embeddings_tsne.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("t-SNE plot saved to %s", path)
        return str(path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engine = GraphMLEngine()
    engine.build_graph()
    engine.train_node2vec(dimensions=32)
    predictions = engine.predict_links(top_k=15)
    print(f"\nTop {len(predictions)} predicted links:")
    for p in predictions[:5]:
        print(f"  {p['source_name'][:30]} ↔ {p['target_name'][:30]} ({p['similarity']:.3f})")
    engine.plot_embeddings()
