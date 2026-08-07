"""Answer Generation Agent.

Generates natural language answers with source attribution and confidence labeling.
Uses the LLM (DeepSeek) to synthesize retrieved context into user-facing responses.

Follows §E.2-3 rules from UrbanGraph-SG-report.md:
- Every factual claim must have [Source: ...] citation
- "I don't know" when context insufficient
- Numerical values must have time range context
- Confidence: HIGH (green) / MEDIUM (yellow) / LOW (orange) / UNKNOWN
"""

import logging
from typing import Any

from src.config import config
from src.graphrag.llm_client import LLMClient
import yaml
from pathlib import Path
from src.retrieval.local_search import search as local_search
from src.retrieval.global_search import search as global_search
from src.retrieval.cypher_agent import execute as cypher_execute

logger = logging.getLogger(__name__)


class AnswerGenerator:
    """Generate answers from retrieved knowledge graph context."""

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient()
        prompt_path = Path(__file__).parent / "prompts" / "answer_prompt.yaml"
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.prompt = yaml.safe_load(f)

    def answer(self, query: str, retrieval_mode: str = "auto") -> dict[str, Any]:
        """Answer a user question about Singapore urban data.

        Args:
            query: User's natural language question
            retrieval_mode: "local", "global", "cypher", or "auto" (tries all)

        Returns:
            dict with answer_text, confidence, sources_used, graph_entities, tokens
        """
        # Step 1: Determine retrieval mode
        if retrieval_mode == "auto":
            retrieval_mode = self._classify_query(query)

        # Step 2: Retrieve context based on mode
        context_data = self._retrieve(query, retrieval_mode)

        # Step 3: Generate answer with LLM
        answer_data = self._generate(query, context_data, retrieval_mode)

        # Step 4: Add confidence labeling
        answer_data["confidence"] = self._assess_confidence(context_data, retrieval_mode)

        return answer_data

    def _classify_query(self, query: str) -> str:
        """Classify query into best retrieval mode."""
        q = query.lower()

        # Cypher candidates: specific listing/aggregation queries
        cypher_patterns = [
            "how many", "list all", "which mrt lines", "find all",
            "what are all", "total number", "count",
        ]
        for pattern in cypher_patterns:
            if pattern in q:
                return "cypher"

        # Global candidates: comparison/summary queries
        global_patterns = [
            "compare", "which area", "which planning", "across",
            "overall", "ranking", "most", "least", "highest", "lowest",
        ]
        for pattern in global_patterns:
            if pattern in q:
                return "global"

        # Default: local search
        return "local"

    def _retrieve(self, query: str, mode: str) -> dict[str, Any]:
        """Retrieve context using the selected mode."""
        if mode == "local":
            return local_search(query)
        elif mode == "global":
            return global_search(query)
        elif mode == "cypher":
            return self._cypher_retrieve(query)
        else:
            return local_search(query)

    def _cypher_retrieve(self, query: str) -> dict[str, Any]:
        """Try to execute query as Cypher. Falls back to local search."""
        # Try to match against preset queries
        from src.retrieval.cypher_agent import PRESET_QUERIES, run_preset

        q = query.lower()

        # Map common queries to presets
        preset_map = {
            "mrt station": "station_count",
            "bishan": "mrt_lines_bishan",
            "jurong east": "lines_at_jurong_east",
            "mrt line": "lines_at_jurong_east",
            "cbd": "mrt_in_cbd",
            "downtown": "mrt_in_cbd",
            "most mrt": "areas_with_most_mrt",
            "bus stop": "bus_stops_orchard",
            "orchard road": "bus_stops_orchard",
            "punggol": "hdb_price_punggol",
        }

        # Population queries: extract area name and use parameterized query
        if "population" in q:
            # Try to find known planning area names in the query
            import pandas as pd
            try:
                pop_df = pd.read_parquet(str(config.data_dir / "raw" / "singstat" / "population.parquet"))
                area_names = pop_df["planning_area"].str.lower().tolist()
            except Exception:
                area_names = []

            for area in area_names:
                if area in q:
                    result = run_preset("planning_area_population", {"area_name": area})
                    if "error" not in result and result.get("results"):
                        return self._format_cypher_result(result, f"Population of {area.title()}")
                    break

        for keyword, preset_id in preset_map.items():
            if keyword in q:
                result = run_preset(preset_id)
                if "error" not in result and result.get("results"):
                    return self._format_cypher_result(result)

        # Fallback to local search
        return local_search(query)

    def _generate(
        self, query: str, context_data: dict[str, Any], mode: str,
    ) -> dict[str, Any]:
        """Call LLM to generate the final answer."""
        context_text = context_data.get("context_text", "No context available.")
        sources = context_data.get("sources", [])
        sources_text = "\n".join(sources[:10])

        system_prompt = self.prompt.get("system", "")
        user_prompt = self.prompt.get("user", "{user_query}")

        user_prompt = user_prompt.replace("{retrieved_subgraph}", context_text[:3000])
        user_prompt = user_prompt.replace("{source_citations}", sources_text[:1000])
        user_prompt = user_prompt.replace("{user_query}", query)

        try:
            answer_text = self.llm.chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=500,
                temperature=0.1,
                label=f"answer_{mode}",
            )
        except RuntimeError as e:
            logger.error("Answer generation failed: %s", e)
            answer_text = "Service temporarily unavailable. Please try again."

        stats = self.llm.get_stats()

        return {
            "answer_text": answer_text.strip(),
            "sources_used": sources[:10],
            "retrieval_mode": mode,
            "llm_model": self.llm.model,
            "tokens_used": stats.get("total_tokens", 0),
            "context_length": len(context_text),
        }

    def _format_cypher_result(self, result: dict[str, Any], label: str = "") -> dict[str, Any]:
        """Format Cypher query results as readable context."""
        cols = result.get("columns", [])
        rows = result["results"]
        prefix = f"{label}: " if label else ""
        readable_lines = [f"{prefix}Cypher query returned {result['count']} result(s)."]
        for i, row in enumerate(rows[:20]):
            parts = []
            for col in cols:
                val = row.get(col, "")
                parts.append(f"{col}: {val}")
            readable_lines.append(f"  {i+1}. " + ", ".join(parts))
        return {
            "entities": [],
            "subgraph": {"nodes": [], "edges": []},
            "context_text": "\n".join(readable_lines),
            "sources": [
                f"[Source: Neo4j graph database, Cypher query, {result['count']} records retrieved, Singapore urban data 2024-2025]"
            ],
        }

    def _assess_confidence(
        self, context_data: dict[str, Any], mode: str,
    ) -> str:
        """Assess answer confidence based on context quality."""
        entities = context_data.get("entities", [])
        subgraph = context_data.get("subgraph", {})
        context_text = context_data.get("context_text", "")

        # No context at all
        if not entities and not context_text:
            return "UNKNOWN"

        # Cypher queries are deterministic
        if mode == "cypher" and "error" not in str(context_data):
            return "HIGH"

        # Context from exact entity matches
        if len(entities) >= 2 and len(subgraph.get("edges", [])) >= 3:
            return "HIGH"

        # Sparse context
        if len(entities) < 2:
            return "MEDIUM"

        return "LOW"


# Standalone convenience function
def answer_query(query: str) -> dict[str, Any]:
    """Answer a single question. Convenience wrapper."""
    gen = AnswerGenerator()
    return gen.answer(query)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    gen = AnswerGenerator()

    # Test a few questions
    for q in [
        "How many MRT stations are there in total?",
        "Which MRT lines pass through Jurong East?",
        "What is the population of Bedok?",
    ]:
        print(f"\nQ: {q}")
        result = gen.answer(q)
        print(f"Mode: {result['retrieval_mode']}, Confidence: {result.get('confidence','?')}")
        print(f"A: {result['answer_text'][:300]}...")
