"""UrbanGraph-SG FastAPI Server.

Endpoints:
  POST /api/query      — Natural language Q&A
  POST /api/cypher     — Execute Cypher query
  GET  /api/stats      — Knowledge graph statistics
  GET  /api/entities   — Semantic entity search
  GET  /api/presets    — List available preset queries

Usage:
  uvicorn src.api.server:app --port 8080
"""

from typing import Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="UrbanGraph-SG API",
    description="GraphRAG-powered Singapore urban knowledge navigator",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ──
class QueryRequest(BaseModel):
    question: str

class CypherRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    confidence: str
    mode: str
    sources: list[str] = []

class CypherResponse(BaseModel):
    query: str
    results: list[dict[str, Any]]
    count: int
    error: str | None = None


# ── Startup ──
@app.on_event("startup")
async def startup():
    from src.graph.neo4j_client import get_driver
    get_driver()


# ── GET /api/stats ──
@app.get("/api/stats")
async def stats():
    try:
        from src.graph.neo4j_client import run_query
        nodes = run_query("MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt ORDER BY cnt DESC")
        rels = run_query("MATCH ()-[r]->() RETURN count(r) AS total")
        return {
            "total_nodes": sum(r["cnt"] for r in nodes),
            "total_edges": rels[0]["total"],
            "nodes_by_type": {r["label"]: r["cnt"] for r in nodes},
        }
    except Exception as e:
        raise HTTPException(500, str(e))


# ── POST /api/query ──
@app.post("/api/query")
async def query(req: QueryRequest):
    try:
        from src.generation.answer_generator import AnswerGenerator
        gen = AnswerGenerator()
        r = gen.answer(req.question)
        return {
            "question": req.question,
            "answer": r["answer_text"],
            "confidence": r.get("confidence", "MEDIUM"),
            "mode": r.get("retrieval_mode", "auto"),
            "sources": r.get("sources_used", []),
        }
    except Exception as e:
        raise HTTPException(500, str(e))


# ── POST /api/cypher ──
@app.post("/api/cypher")
async def cypher(req: CypherRequest):
    try:
        from src.retrieval.cypher_agent import execute
        r = execute(req.query)
        if "error" in r:
            return {"query": req.query, "results": [], "count": 0, "error": r["error"]}
        return {
            "query": req.query,
            "results": r["results"],
            "count": r["count"],
            "error": None,
        }
    except Exception as e:
        raise HTTPException(500, str(e))


# ── GET /api/entities ──
@app.get("/api/entities")
async def entities(q: str = "", limit: int = 10):
    try:
        from src.retrieval.semantic_search import get_engine
        engine = get_engine()
        engine.load()
        results = engine.search(q, top_k=limit)
        return {"query": q, "results": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(500, str(e))


# ── GET /api/presets ──
@app.get("/api/presets")
async def presets():
    from src.retrieval.cypher_agent import list_presets
    return {"presets": sorted(list_presets())}


# ── Health check ──
@app.get("/api/health")
async def health():
    return {"status": "healthy", "service": "UrbanGraph-SG API v2.0.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8080, reload=True)
