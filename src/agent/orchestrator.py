"""Multi-Agent Orchestrator.

Architecture: PlannerAgent -> RetrieverAgent -> ReasonerAgent -> VisualizerAgent

Each agent has a specialized role:
- Planner: Classifies query and selects retrieval strategy
- Retriever: Executes Cypher / semantic / local search
- Reasoner: Calls LLM to synthesize answer from retrieved context
- Visualizer: Maps answer entities to map coordinates
"""

import logging
from typing import Any
from dataclasses import dataclass

from src.graph.neo4j_client import run_query

logger = logging.getLogger(__name__)


@dataclass
class AgentContext:
    query: str
    retrieval_mode: str = "auto"
    entities: list[dict] = None
    context_text: str = ""
    sources: list[str] = None
    answer: str = ""
    confidence: str = "MEDIUM"
    highlights: list[dict] = None

    def __post_init__(self):
        self.entities = self.entities or []
        self.sources = self.sources or []
        self.highlights = self.highlights or []


class PlannerAgent:
    """Classify query intent and select retrieval strategy."""

    def plan(self, query: str) -> str:
        q = query.lower()
        if any(k in q for k in ["how many", "total", "count", "list", "which line", "what line"]):
            return "cypher"
        if any(k in q for k in ["compare", "most", "least", "highest", "lowest", "ranking"]):
            return "global"
        return "semantic"


class RetrieverAgent:
    """Execute retrieval: cypher -> semantic -> local."""

    def retrieve(self, ctx: AgentContext) -> AgentContext:
        query = ctx.query

        # Try Cypher first (direct preset)
        try:
            from src.generation.answer_generator import AnswerGenerator
            gen = AnswerGenerator()
            direct = gen._try_direct_preset(query)
            if direct:
                ctx.context_text = direct.get("context_text", "")
                ctx.sources = direct.get("sources_used", [])
                ctx.retrieval_mode = "cypher"
                ctx.answer = direct.get("answer_text", "")
                ctx.confidence = "HIGH"
                return ctx
        except Exception as e:
            logger.debug("Direct preset failed: %s", e)

        # Try ChromaDB semantic
        try:
            from src.retrieval.vector_store import get_store
            sem = get_store().search_and_format(query)
            if sem and sem.get("entities"):
                ctx.entities = sem["entities"]
                ctx.context_text = sem["context_text"]
                ctx.sources = sem["sources"]
                ctx.retrieval_mode = "semantic"
                return ctx
        except Exception as e:
            logger.debug("Semantic search failed: %s", e)

        # Fallback: local search
        try:
            from src.retrieval.local_search import search
            local = search(query)
            ctx.entities = local.get("entities", [])
            ctx.context_text = local.get("context_text", "")
            ctx.sources = local.get("sources", [])
            ctx.retrieval_mode = "local"
        except Exception as e:
            logger.debug("Local search failed: %s", e)

        return ctx


class ReasonerAgent:
    """Call LLM to generate final answer."""

    def reason(self, ctx: AgentContext) -> AgentContext:
        if ctx.answer:
            return ctx  # already answered by direct preset

        try:
            from src.generation.answer_generator import AnswerGenerator
            gen = AnswerGenerator()
            context_data = {
                "context_text": ctx.context_text,
                "sources": ctx.sources,
                "entities": ctx.entities,
                "subgraph": {"nodes": [], "edges": []},
            }
            result = gen._generate(ctx.query, context_data, ctx.retrieval_mode)
            ctx.answer = result.get("answer_text", "I don't have enough data.")
            ctx.confidence = result.get("confidence", "MEDIUM")
        except Exception as e:
            ctx.answer = f"Service temporarily unavailable. ({e})"
            ctx.confidence = "LOW"

        return ctx


class VisualizerAgent:
    """Map answer entities to map coordinates for highlighting."""

    def visualize(self, ctx: AgentContext) -> AgentContext:
        if ctx.highlights:
            return ctx

        # Extract entity mentions from answer and find coordinates
        try:
            names = run_query("MATCH (n) WHERE n.name IS NOT NULL AND n.lat IS NOT NULL RETURN DISTINCT n.name AS name, n.lat AS lat, n.lon AS lon, labels(n)[0] AS label LIMIT 6000")
            for n in names:
                name = str(n.get("name", ""))
                if len(name) > 3 and name.lower() in ctx.answer.lower():
                    lat = n.get("lat")
                    lon = n.get("lon")
                    if lat and lon and len(ctx.highlights) < 10:
                        ctx.highlights.append({"name": name, "lat": float(lat), "lon": float(lon), "label": n.get("label", "")})
        except Exception:
            pass

        return ctx


class Orchestrator:
    """Coordinate multi-agent pipeline."""

    def __init__(self):
        self.planner = PlannerAgent()
        self.retriever = RetrieverAgent()
        self.reasoner = ReasonerAgent()
        self.visualizer = VisualizerAgent()

    def process(self, query: str) -> AgentContext:
        ctx = AgentContext(query=query)
        ctx.retrieval_mode = self.planner.plan(query)
        ctx = self.retriever.retrieve(ctx)
        ctx = self.reasoner.reason(ctx)
        ctx = self.visualizer.visualize(ctx)
        return ctx


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    orch = Orchestrator()
    for q in ["How many MRT stations are there?", "What weather stations are near Changi?"]:
        ctx = orch.process(q)
        print(f"[{ctx.retrieval_mode}/{ctx.confidence}] {q}")
        print(f"  A: {ctx.answer[:150]}...")
        print(f"  Highlights: {[h['name'] for h in ctx.highlights]}")
        print()
