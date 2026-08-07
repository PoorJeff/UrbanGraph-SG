"""Semantic Entity Search using text embeddings (all-MiniLM-L6-v2).

Replaces brittle keyword matching with dense vector similarity.
Loads 5,500+ entities from Neo4j, builds embeddings, searches by cosine similarity.
"""

import logging
from typing import Any
import numpy as np

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False

class SemanticSearchEngine:
    def __init__(self):
        self.model = None
        self.texts: list[str] = []
        self.ids: list[str] = []
        self.data: list[dict] = []
        self.embeddings: np.ndarray | None = None
        self._loaded = False

    def load(self, force=False) -> bool:
        if self._loaded and not force: return True
        if not HAS_SBERT: return False
        try:
            from src.graph.neo4j_client import run_query
            rows = run_query("""MATCH (n) WHERE n.name IS NOT NULL
                RETURN n.id AS id, n.name AS name, n.description AS d, labels(n)[0] AS l, n.lat AS lat, n.lon AS lon LIMIT 6000""")
        except Exception as e:
            logger.warning("Neo4j load failed: %s", e); return False
        if not rows: return False

        self.texts, self.ids, self.data = [], [], []
        for r in rows:
            nm = str(r.get("name","")); lb = str(r.get("l",""))
            self.texts.append(f"{nm}. {str(r.get('d',''))[:150]}. {lb}")
            self.ids.append(str(r.get("id","")))
            self.data.append({"id":str(r.get("id","")),"name":nm,"label":lb,"lat":r.get("lat"),"lon":r.get("lon")})
        try:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.embeddings = self.model.encode(self.texts, show_progress_bar=False, batch_size=256)
            self._loaded = True
            logger.info("Semantic search ready: %d entities, emb shape %s", len(self.texts), self.embeddings.shape)
            return True
        except Exception as e:
            logger.error("Embedding failed: %s", e); return False

    def search(self, query: str, top_k=5) -> list[dict]:
        if not self._loaded and not self.load(): return []
        try:
            qe = self.model.encode([query], show_progress_bar=False)[0]
            sims = np.dot(self.embeddings, qe) / (np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(qe) + 1e-10)
            results = []
            for idx in np.argsort(sims)[::-1][:top_k*2]:
                s = float(sims[idx])
                if s < 0.35 or len(results) >= top_k: continue
                e = dict(self.data[idx]); e["similarity"] = round(s, 3); results.append(e)
            return results
        except Exception as e:
            logger.error("Search failed: %s", e); return []

    def search_and_format(self, query: str, top_k=5) -> dict[str, Any]:
        results = self.search(query, top_k)
        if not results: return {"entities":[],"subgraph":{"nodes":[],"edges":[]},"context_text":"","sources":[]}
        lines = ["Semantic search results (dense vector similarity):"]
        entities, sources = [], []
        for i, r in enumerate(results):
            lines.append(f"  {i+1}. [{r['label']}] {r['name']} (score: {r['similarity']:.2f})")
            sources.append(f"[Source: {r['name']}, id={r['id']}, similarity={r['similarity']}]")
            entities.append({"name":r["name"],"id":r["id"],"labels":[r["label"]],"lat":r.get("lat"),"lon":r.get("lon"),"similarity":r["similarity"]})
        return {"entities":entities,"subgraph":{"nodes":[],"edges":[]},"context_text":"\n".join(lines),"sources":sources}

_engine = None
def get_engine(): global _engine; _engine = _engine or SemanticSearchEngine(); return _engine
