"""ChromaDB Vector Store for entity embeddings.

Persists sentence-transformer embeddings to disk so they survive restarts.
Enables hybrid search: graph (Neo4j Cypher) + vector (ChromaDB).

Usage:
  store = get_store()
  store.index_entities()          # Build index once
  results = store.search("query") # Semantic search
"""

import logging, json
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False

from src.config import config

STORE_DIR = Path(config.settings.data_dir).parent / "chroma_db"


class VectorStore:
    def __init__(self):
        self.client = None
        self.collection = None
        self._ready = False

    def _ensure_client(self):
        if not HAS_CHROMA:
            return False
        if self.client is None:
            try:
                STORE_DIR.mkdir(parents=True, exist_ok=True)
                self.client = chromadb.PersistentClient(path=str(STORE_DIR))
                self.collection = self.client.get_or_create_collection(
                    name="urbangraph_entities",
                    metadata={"hnsw:space": "cosine"},
                )
                self._ready = True
            except Exception as e:
                logger.error("ChromaDB init failed: %s", e)
                return False
        return True

    def index_entities(self, force=False) -> int:
        """Load all entities from Neo4j and index into ChromaDB.

        Returns number of entities indexed.
        """
        if not self._ensure_client():
            return 0

        # Skip if already indexed (unless forced)
        if not force and self.collection and self.collection.count() > 1000:
            logger.info("Vector store already has %d entities", self.collection.count())
            return self.collection.count()

        try:
            from src.graph.neo4j_client import run_query
            from sentence_transformers import SentenceTransformer

            rows = run_query("""MATCH (n) WHERE n.name IS NOT NULL
                RETURN n.id AS id, n.name AS name, n.description AS d,
                       labels(n)[0] AS l, n.lat AS lat, n.lon AS lon LIMIT 6000""")
        except Exception as e:
            logger.error("Neo4j load failed: %s", e)
            return 0

        if not rows:
            return 0

        model = SentenceTransformer("all-MiniLM-L6-v2")
        texts, ids, metadatas, embeddings_list = [], [], [], []

        for r in rows:
            eid = str(r.get("id", ""))
            name = str(r.get("name", ""))
            label = str(r.get("l", ""))
            desc = str(r.get("d", ""))[:200]
            lat = r.get("lat")
            lon = r.get("lon")

            text = f"{name}. {desc}. Type: {label}"
            texts.append(text)
            ids.append(eid)
            metadatas.append({
                "name": name, "label": label, "desc": desc[:100],
                "lat": float(lat) if lat else 0,
                "lon": float(lon) if lon else 0,
            })

        # Batch embed
        logger.info("Embedding %d entities...", len(texts))
        embeddings = model.encode(texts, show_progress_bar=False, batch_size=256)

        # Upsert to ChromaDB in batches
        batch_size = 500
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            self.collection.upsert(
                ids=ids[i:end],
                embeddings=embeddings[i:end].tolist(),
                metadatas=metadatas[i:end],
                documents=texts[i:end],
            )

        logger.info("Vector store: indexed %d entities", len(ids))
        return len(ids)

    def search(self, query: str, top_k=5) -> list[dict[str, Any]]:
        """Semantic search via ChromaDB."""
        if not self._ensure_client() or not self.collection:
            return []

        # Re-index if empty
        if self.collection.count() < 100:
            self.index_entities()

        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            qe = model.encode([query], show_progress_bar=False)[0].tolist()

            results = self.collection.query(
                query_embeddings=[qe],
                n_results=top_k,
            )

            entities = []
            if results and results.get("ids") and results["ids"][0]:
                for i, eid in enumerate(results["ids"][0]):
                    meta = results["metadatas"][0][i] if results.get("metadatas") else {}
                    dist = results["distances"][0][i] if results.get("distances") else 0
                    entities.append({
                        "id": eid,
                        "name": meta.get("name", ""),
                        "label": meta.get("label", ""),
                        "lat": meta.get("lat"),
                        "lon": meta.get("lon"),
                        "score": round(1 - dist, 3) if dist else 0,
                    })
            return entities
        except Exception as e:
            logger.error("Vector search failed: %s", e)
            return []

    def search_and_format(self, query: str, top_k=5) -> dict[str, Any]:
        """Search and format for answer generation."""
        results = self.search(query, top_k)
        if not results:
            return {"entities": [], "subgraph": {"nodes":[],"edges":[]}, "context_text": "", "sources": []}

        lines = ["Vector search results (ChromaDB + all-MiniLM-L6-v2):"]
        entities, sources = [], []
        for i, r in enumerate(results):
            lines.append(f"  {i+1}. [{r['label']}] {r['name']} (score: {r['score']})")
            sources.append(f"[Source: {r['name']}, id={r['id']}, score={r['score']}]")
            entities.append({"name": r["name"], "id": r["id"], "labels": [r["label"]],
                           "lat": r.get("lat"), "lon": r.get("lon")})
        return {"entities": entities, "subgraph": {"nodes":[],"edges":[]},
                "context_text": "\n".join(lines), "sources": sources}


_store = None
def get_store() -> VectorStore:
    global _store
    _store = _store or VectorStore()
    return _store
