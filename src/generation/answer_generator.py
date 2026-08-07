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
        self._context_cache: dict[str, Any] = {}  # multi-turn memory

    def answer(self, query: str, retrieval_mode: str = "auto", context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Answer a user question about Singapore urban data.

        Args:
            query: User's natural language question
            retrieval_mode: "local", "global", "cypher", or "auto" (tries all)

        Returns:
            dict with answer_text, confidence, sources_used, graph_entities, tokens
            Also returns 'context' dict for multi-turn conversation memory.
        """
        # Step 0: Multi-turn context injection
        merged_query = query
        effective_context = context or self._context_cache
        if effective_context and effective_context.get("intent") and effective_context.get("entities"):
            merged, _ = self._merge_context(query, effective_context)
            if merged and merged != query:
                merged_query = merged

        # Step 1: Direct preset mapping (exact known questions)
        direct = self._try_direct_preset(merged_query)
        if direct:
            self._save_context(direct, merged_query)
            return direct

        # Step 1.5: NEW 4-layer query parser (NER + Intent + Slot Fill + Execute)
        try:
            from src.retrieval.query_parser import parse_and_execute
            parsed = parse_and_execute(merged_query)
            if parsed and parsed.get("results"):
                data = self._format_cypher_result(parsed, f"Parsed: {parsed.get('template','?')}")
                answer = self._generate(merged_query, data, "cypher")
                answer["confidence"] = "HIGH"
                self._save_context(answer, merged_query)
                return answer
        except Exception:
            pass

        # Step 2: Cypher keyword matching (fallback) — also saves context
        from src.retrieval.cypher_agent import run_preset
        q = query.lower()
        cypher_data = None

        # Try keyword → preset matching (simplified, no full _cypher_retrieve)
        for keyword, preset_id in [
            ("station_count", ["mrt station", "total mrt"]),
            ("bus_stop_count", ["bus stop", "total bus"]),
            ("circle_line_stations", ["circle line"]),
            ("mrt_count_cbd", ["cbd", "downtown"]),
            ("areas_with_most_mrt", ["most mrt", "highest mrt count"]),
            ("areas_with_least_mrt", ["least mrt", "fewest mrt"]),
            ("mrt_lines_bishan", ["bishan"]),
            ("lines_at_jurong_east", ["jurong east"]),
            ("lines_at_woodlands", ["woodlands"]),
            ("stations_most_connections", ["most connection", "most interchange"]),
            ("hdb_highest_prices", ["hdb price", "most expensive", "highest hdb"]),
            ("largest_population", ["largest population", "most populated"]),
            ("smallest_population", ["smallest population", "least populated"]),
        ]:
            if keyword == "station_count" and ("cbd" in q or "downtown" in q or "circle" in q):
                continue  # skip generic when specific area mentioned
            if keyword == "bus_stop_count" and ("orchard road" in q):
                continue
            if any(p in q for p in preset_id):
                r = run_preset(keyword)
                if r.get("results"):
                    data = self._format_cypher_result(r)
                    answer = self._generate(query, data, "cypher")
                    answer["confidence"] = "HIGH"
                    return answer

        # Step 3: Semantic search (embedding-based, two backends)
        semantic_data = None
        try:
            from src.retrieval.vector_store import get_store
            store = get_store()
            semantic_data = store.search_and_format(query)
        except Exception:
            pass
        if not semantic_data or not semantic_data.get("entities"):
            try:
                from src.retrieval.semantic_search import get_engine
                semantic_data = get_engine().search_and_format(query)
            except Exception:
                pass
        if semantic_data and semantic_data.get("entities"):
            answer = self._generate(query, semantic_data, "semantic")
            answer["confidence"] = "MEDIUM"
            return answer

        # Step 4: Local search as last resort
        context_data = local_search(query)
        answer = self._generate(merged_query, context_data, "local")
        answer["confidence"] = self._assess_confidence(context_data, "local")
        if next_ctx: answer["next_context"] = next_ctx
        return answer

    def _merge_context(self, query: str, ctx: dict) -> tuple[str, dict]:
        """Merge previous conversation context into the current query."""
        prev_intent = ctx.get("intent", "")
        prev_entities = ctx.get("entities", [])
        q = query.lower().strip()

        # Follow-up detection
        followup_markers = ["what about", "and what about", "how about",
                           "那", "and", "also", "too"]
        is_followup = any(m in q for m in followup_markers) or len(q.split()) <= 3

        if not is_followup or not prev_intent:
            return query, {}

        # Extract new entity from follow-up query
        from src.retrieval.query_parser import link_entities
        linked = link_entities(query)
        new_entities = [e for e in linked["entities"] if e.get("confidence", 0) > 0.8]

        if not new_entities:
            return query, {}

        # Build merged query: "previous_intent for new_entity"
        new_name = new_entities[0]["name"]
        intent_templates = {
            "DESCRIBE": f"What is the population of {new_name}?",
            "LIST": f"Which MRT lines pass through {new_name}?",
            "COUNT": f"How many MRT stations are in {new_name}?",
            "LOCATE": f"Which planning area is {new_name} in?",
        }
        merged = intent_templates.get(prev_intent, query)
        next_ctx = {"intent": prev_intent, "entities": [{"name": new_name}]}
        return merged, next_ctx

    def _save_context(self, result: dict[str, Any], query: str):
        """Save context for next turn."""
        try:
            from src.retrieval.query_parser import link_entities, classify_intent
            linked = link_entities(query)
            intent = classify_intent(query, linked["entities"])
            self._context_cache = {
                "intent": intent.get("intent", ""),
                "entities": [{"name": e["name"]} for e in linked["entities"][:3]],
                "query": query,
            }
        except Exception:
            self._context_cache = {}

    def _try_direct_preset(self, query: str) -> dict[str, Any] | None:
        """Direct preset lookup. Bypasses ALL routing logic for reliability."""
        from src.retrieval.cypher_agent import run_preset

        q = query.lower().strip()
        cypher_data = None

        # MRT counts — specific areas first, then generic
        if "cbd" in q and "mrt" in q:
            r = run_preset("mrt_count_cbd")
            if r.get("results"):
                cypher_data = self._format_cypher_result(r, "MRT in CBD")
        elif "circle line" in q:
            r = run_preset("circle_line_stations")
            if r.get("results"):
                cypher_data = self._format_cypher_result(r, "Circle Line stations")
        elif ("how many mrt" in q and "total" not in q and "cbd" not in q) or ("mrt station" in q and "total" in q):
            r = run_preset("station_count")
            if r.get("results"):
                cypher_data = self._format_cypher_result(r, "Total MRT stations")

        # Bus counts
        elif "bus stop" in q and ("how many" in q or "total" in q):
            r = run_preset("bus_stop_count")
            if r.get("results"):
                cypher_data = self._format_cypher_result(r, "Total bus stops")

        # MRT lines at station
        elif "lines" in q and "bishan" in q:
            r = run_preset("mrt_lines_bishan")
            if r.get("results"):
                cypher_data = self._format_cypher_result(r, "Lines at Bishan")
        elif "lines" in q and "jurong east" in q:
            r = run_preset("lines_at_jurong_east")
            if r.get("results"):
                cypher_data = self._format_cypher_result(r, "Lines at Jurong East")
        elif "lines" in q and "woodlands" in q:
            r = run_preset("lines_at_woodlands")
            if r.get("results"):
                cypher_data = self._format_cypher_result(r, "Lines at Woodlands")

        # MRT in area (listing)
        elif "mrt station" in q and ("in" in q or "list" in q):
            import re
            m = re.search(r'(?:in|list)\s+(?:all\s+)?(?:mrt\s+(?:stations?\s+)?)?(?:in\s+)?(\w[\w\s]+?)(?:\s+area|\s+planning|\s*$|\s*\?)', q)
            if m:
                area = m.group(1).strip()
                if area:
                    r = run_preset("mrt_in_any_area", {"area": area})
                    if r.get("results"):
                        cypher_data = self._format_cypher_result(r, f"MRT in {area}")

        # MRT by area ranking
        elif "most mrt" in q or "highest mrt" in q:
            r = run_preset("areas_with_most_mrt")
            if r.get("results"):
                cypher_data = self._format_cypher_result(r, "Most MRT areas")
        elif ("least mrt" in q or "fewest mrt" in q or "smallest" in q and "mrt" in q):
            r = run_preset("areas_with_least_mrt")
            if r.get("results"):
                cypher_data = self._format_cypher_result(r, "Least MRT areas")

        # Connections/Path — specific patterns first
        elif "most connection" in q:
            r = run_preset("stations_most_connections")
            if r.get("results"):
                cypher_data = self._format_cypher_result(r, "Most connected stations")
        elif "connected" in q or ("stations" in q and ("from" in q or "to" in q)):
            if "bishan" in q and "orchard" in q:
                r = run_preset("bishan_to_orchard_path")
                if r.get("results"):
                    cypher_data = self._format_cypher_result(r, "Bishan→Orchard")
            elif "jurong east" in q and "city hall" in q:
                r = run_preset("jurong_east_to_city_hall_path")
                if r.get("results"):
                    cypher_data = self._format_cypher_result(r, "Jurong East→City Hall")

        # Population
        elif "population" in q:
            import re
            m = re.search(r'(?:population\s+(?:of\s+)?|of\s+)(\w[\w\s]+?)(?:\s*$|\s*\?|\s+planning)', q)
            if not m:
                m = re.search(r'(\w+)(?:\'s\s+population|\s+population)', q)
            if m:
                area = m.group(1).strip()
                r = run_preset("planning_area_population", {"area_name": area})
                if r.get("results"):
                    cypher_data = self._format_cypher_result(r, f"Population of {area}")
        elif "largest population" in q or "highest population" in q or "most populated" in q:
            r = run_preset("largest_population")
            if r.get("results"):
                cypher_data = self._format_cypher_result(r, "Largest population")
        elif "smallest population" in q or "lowest population" in q or "least populated" in q:
            r = run_preset("smallest_population")
            if r.get("results"):
                cypher_data = self._format_cypher_result(r, "Smallest population")

        # HDB
        elif "highest hdb" in q or "hdb price" in q or "expensive" in q:
            r = run_preset("hdb_highest_prices")
            if r.get("results"):
                cypher_data = self._format_cypher_result(r, "Highest HDB prices")
        elif "hdb transaction" in q:
            r = run_preset("hdb_total_transactions")
            if r.get("results"):
                cypher_data = self._format_cypher_result(r, "Total HDB transactions")

        # Planning area lookup
        elif "planning area" in q and "in" in q:
            import re
            m = re.search(r'(?:in|is)\s+(\w[\w\s]+?)(?:\s+mrt|\s+station|\s*$|\s*\?)', q)
            if m:
                stn = m.group(1).strip()
                r = run_preset("station_area_lookup", {"station": stn})
                if r.get("results"):
                    cypher_data = self._format_cypher_result(r, f"Area of {stn}")

        # Bus near station
        elif "bus stop" in q and "near" in q and "mrt" in q:
            import re
            m = re.search(r'near\s+(\w[\w\s]+?)(?:\s+mrt|\s*$)', q)
            if m:
                stn = m.group(1).strip()
                r = run_preset("bus_near_station", {"station": stn})
                if r.get("results"):
                    cypher_data = self._format_cypher_result(r, f"Bus stops near {stn}")

        # Bus stops on road
        elif "bus stop" in q and "orchard road" in q:
            r = run_preset("bus_stops_orchard")
            if r.get("results"):
                cypher_data = self._format_cypher_result(r, "Bus stops on Orchard Road")

        if cypher_data:
            answer_data = self._generate(query, cypher_data, "cypher")
            answer_data["confidence"] = "HIGH"
            return answer_data
        return None

    def _classify_query(self, query: str) -> str:
        """Classify query into best retrieval mode."""
        q = query.lower()

        # Cypher candidates: specific listing/aggregation queries
        cypher_patterns = [
            "how many", "list all", "which mrt lines", "find all",
            "what are all", "total number", "count", "most mrt",
            "population of", "population",
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

    def _retrieve_semantic(self, query: str, mode: str) -> dict[str, Any]:
        """Fallback: semantic search (local or global)."""
        if mode == "global":
            return global_search(query)
        else:
            return local_search(query)

    def _cypher_retrieve(self, query: str) -> dict[str, Any]:
        """Try to execute query as Cypher. Falls back to local search."""
        import re
        from src.retrieval.cypher_agent import PRESET_QUERIES, run_preset

        q = query.lower()

        # Map common queries to presets
        preset_map = [
            # ── SPECIFIC queries first ──
            ("largest population", "largest_population"),
            ("highest population", "largest_population"),
            ("smallest population", "smallest_population"),
            ("lowest population", "smallest_population"),
            ("most connections", "stations_most_connections"),
            ("most mrt", "areas_with_most_mrt"),
            ("least mrt", "areas_with_least_mrt"),
            ("fewest mrt", "areas_with_least_mrt"),
            ("highest hdb", "hdb_highest_prices"),
            ("expensive", "hdb_highest_prices"),
            ("hdb transaction", "hdb_total_transactions"),
            ("cbd", "mrt_count_cbd"),
            ("downtown core", "mrt_count_cbd"),
            ("circle line", "circle_line_stations"),
            ("bishan", "mrt_lines_bishan"),
            ("jurong east", "lines_at_jurong_east"),
            ("woodlands", "lines_at_woodlands"),
            ("city hall", "lines_at_city_hall"),
            ("orchard road", "bus_stops_orchard"),
            ("orchard", "mrt_in_orchard"),  # catch-all for "Orchard MRT", "MRT in Orchard"
            ("hdb price", "hdb_highest_prices"),
            ("punggol", "hdb_price_town"),
            ("hdb resale", "hdb_highest_prices"),
            ("bus service", "bus_stop_count"),
            ("holiday", "next_holiday"),
            ("weather station", "weather_stations"),
            # ── GENERIC last ──
            ("bus stop", "bus_stop_count"),
            ("mrt station", "station_count"),
            ("mrt line", "lines_at_jurong_east"),
        ]

        # Connected/path queries: "is X connected to Y?" or "stations from X to Y"
        if "connected" in q or ("stations" in q and ("from" in q or "to" in q or "between" in q)):
            # Try Bishan-Orchard specific
            if "bishan" in q and "orchard" in q:
                result = run_preset("bishan_to_orchard_path")
                if "error" not in result and result.get("results"):
                    return self._format_cypher_result(result, "Path Bishan→Orchard")
            # Try Jurong East-City Hall specific
            elif "jurong east" in q and "city hall" in q:
                result = run_preset("jurong_east_to_city_hall_path")
                if "error" not in result and result.get("results"):
                    return self._format_cypher_result(result, "Path Jurong East→City Hall")

        # Area lookup: "what planning area is X in?" or "what area is X in"
        if ("planning area" in q or "which area" in q) and " mrt " in q:
            station_match = re.search(r'(?:is|in|area\s+)\s+(\w[\w\s]+?)(?:\s+mrt|\s+station|\s*$|\s*\?)', q)
            if not station_match:
                station_match = re.search(r'(\w+)(?:\s+mrt|\s+station)', q)
            if station_match:
                stn = station_match.group(1).strip()
                result = run_preset("station_area_lookup", {"station": stn})
                if "error" not in result and result.get("results"):
                    return self._format_cypher_result(result, f"Location of {stn}")

        # MRT in area: "mrt stations in X" or "list all mrt in X"
        if ("mrt" in q) and (" in " in q or " list " in q):
            parts = q.split(" in ")
            if len(parts) >= 2:
                area = parts[-1].strip().rstrip("?")
                # Clean up: remove trailing words like "area", "region"
                area = re.sub(r'\s+(area|region|planning)\s*$', '', area)
                area = re.sub(r'[^\w\s]', '', area).strip()
                if area and area not in ("total", "singapore", "all", "the"):
                    result = run_preset("mrt_in_any_area", {"area": area})
                    if "error" not in result and result.get("results"):
                        return self._format_cypher_result(result, f"MRT in {area}")

        # "bus stops near X" or "how many bus stops near X"
        if ("bus stop" in q or "bus station" in q) and ("near" in q or "close to" in q):
            station_match = re.search(r'(?:near|close\s+to)\s+(\w[\w\s]+?)(?:\s+mrt|\s+station|\s*$|\s*\?)', q)
            if station_match:
                stn = station_match.group(1).strip()
                result = run_preset("bus_near_station", {"station": stn})
                if "error" not in result and result.get("results"):
                    return self._format_cypher_result(result, f"Bus stops near {stn}")

        # Population queries: extract area name from query pattern
        if "population" in q:
            # Pattern: "population of X" or "X population"
            pop_match = re.search(r'population\s+of\s+(\w[\w\s]*)', q)
            if not pop_match:
                pop_match = re.search(r'(\w[\w\s]*)\s+population', q)
            if pop_match:
                area_name = pop_match.group(1).strip()
                result = run_preset("planning_area_population", {"area_name": area_name})
                if "error" not in result and result.get("results"):
                    return self._format_cypher_result(result, f"Population of {area_name.title()}")

        for keyword, preset_id in preset_map:
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

        # Replace ALL placeholders correctly:
        # - {retrieved_subgraph} and {source_citations} → in system prompt
        # - {user_query} → in user prompt
        system_prompt = system_prompt.replace("{retrieved_subgraph}", context_text[:3000])
        system_prompt = system_prompt.replace("{source_citations}", sources_text[:1000])
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
        """Format Cypher query results as readable context.

        The context is designed to give the LLM clear, actionable data so it can
        directly answer without needing to interpret complex structures.
        """
        cols = result.get("columns", [])
        rows = result["results"]
        prefix = f"{label}: " if label else ""

        readable_lines = [
            f"Cypher database query returned {result['count']} result(s).",
            f"Use this data to answer the user's question:",
        ]

        # Format as a clear table-like structure
        for i, row in enumerate(rows[:20]):
            parts = []
            for col in cols:
                val = row.get(col, "")
                parts.append(f"{col}={val}")
            readable_lines.append(f"  Result {i+1}: " + ", ".join(parts))

        # Path explanation: expand stations array into readable route
        if "stations" in cols and result['count'] >= 1:
            stations = rows[0].get("stations", [])
            hops = rows[0].get("hops", "?")
            if isinstance(stations, list) and len(stations) > 1:
                path_str = " → ".join(str(s) for s in stations)
                readable_lines.append(f"  Route ({hops} hops): {path_str}")

        # If result is a single count, make it extra explicit
        if result['count'] == 1 and len(cols) == 1:
            col = cols[0]
            val = rows[0].get(col, "?")
            readable_lines.append(f"  --> The answer is: {val}")

        return {
            "entities": [],
            "subgraph": {"nodes": [], "edges": []},
            "context_text": "\n".join(readable_lines),
            "sources": [
                f"[Source: Neo4j graph database, {result['count']} record(s), Singapore urban data 2024-2025]"
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
