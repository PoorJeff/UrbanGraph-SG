"""
Four-layer query understanding pipeline.

Layer 1: Entity Linker  — name → entity_id via exact/normalized/fuzzy/vector
Layer 2: Intent Classifier  — LLM or regex → COUNT/LIST/COMPARE/PATH/LOCATE/DESCRIBE/RANK/UNKNOWN
Layer 3: Slot Filler       — intent + entities → template + params
Layer 4: Validator+Executor — validate → run Cypher → format result

Fallback chain: Cypher → semantic search → local search → "I don't know"
"""

import logging, re, time
from typing import Any

from src.graph.neo4j_client import run_query

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
# ENTITY LOADER (shared memory index)
# ═══════════════════════════════════════════════════════════
_ENTITY_INDEX: dict[str, dict] = {}
_ENTITY_NAMES_LOWER: dict[str, list[dict]] = {}
_LOADED = False


def _load_entity_index():
    global _ENTITY_INDEX, _ENTITY_NAMES_LOWER, _LOADED
    if _LOADED: return
    t0 = time.time()
    rows = run_query("""MATCH (n) WHERE n.name IS NOT NULL
        RETURN n.id AS id, n.name AS name, labels(n)[0] AS label,
               n.lat AS lat, n.lon AS lon, n.transport_type AS tt
        LIMIT 6000""")
    for r in rows:
        eid = str(r["id"]); name = str(r["name"])
        entry = {"id":eid,"name":name,"label":r.get("label",""),"lat":r.get("lat"),"lon":r.get("lon"),"tt":r.get("tt","")}
        _ENTITY_INDEX[eid] = entry
        key = name.lower()
        _ENTITY_NAMES_LOWER.setdefault(key, []).append(entry)
    _LOADED = True
    logger.info("Entity index loaded: %d entities in %.0fms", len(_ENTITY_INDEX), (time.time()-t0)*1000)


def get_entity_index() -> dict[str, dict]:
    """Return the in-memory entity index for highlight matching."""
    _load_entity_index()
    return dict(_ENTITY_INDEX)


def _normalize(name: str) -> str:
    n = name.lower().strip()
    for s in ["mrt station","lrt station","mrt","lrt","station","bus stop","terminal","interchange","stn","int"]:
        n = re.sub(r'\b'+re.escape(s)+r'\b', '', n)
    n = re.sub(r'[^\w\s]', '', n)
    return re.sub(r'\s+', ' ', n).strip()


def _levenshtein(s1: str, s2: str) -> int:
    if len(s1) < len(s2): return _levenshtein(s2, s1)
    if not s2: return len(s1)
    prev = list(range(len(s2)+1))
    for i, c1 in enumerate(s1):
        cur = [i+1]
        for j, c2 in enumerate(s2):
            cur.append(min(cur[-1]+1, prev[j+1]+1, prev[j]+(0 if c1==c2 else 1)))
        prev = cur
    return prev[-1]


# ═══════════════════════════════════════════════════════════
# LAYER 1: ENTITY LINKER
# ═══════════════════════════════════════════════════════════
def link_entities(query: str) -> dict[str, Any]:
    """Extract entity mentions from query and link to Neo4j IDs.

    Returns {entities, unmatched_tokens, ambiguous}
    """
    _load_entity_index()
    # Tokenize: extract capitalized words + known MRT line patterns
    raw_tokens = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|\w+', query)
    # Also try sliding window for multi-word names
    tokens = []
    for t in raw_tokens:
        t = t.strip().rstrip(",?!.")
        if len(t) >= 2 and t.lower() not in ('the','are','for','and','not','how','what','which','from','list','all','any','has','does','many','much','that','this','with','there','their','they','where','when','why','who','whom','can','will','would','should','could','may','might','shall','must','each','every','some','few','most','more','less','very','too','just','also','only','even','then','than'):
            tokens.append(t)

    found = []
    unmatched = []
    seen_ids = set()

    for token in tokens:
        t_lower = token.lower()
        # 1A: Exact match (case-insensitive)
        if t_lower in _ENTITY_NAMES_LOWER:
            for e in _ENTITY_NAMES_LOWER[t_lower]:
                if e["id"] not in seen_ids:
                    found.append({**e, "match_level": "exact", "confidence": 1.0})
                    seen_ids.add(e["id"])
            continue

        # 1B: Normalized match
        norm = _normalize(token)
        if norm and len(norm) >= 2:
            # Try direct lookup
            if norm in _ENTITY_NAMES_LOWER:
                for e in _ENTITY_NAMES_LOWER[norm]:
                    if e["id"] not in seen_ids:
                        found.append({**e, "match_level": "normalized", "confidence": 0.95})
                        seen_ids.add(e["id"])
                continue
            # Try substring match against all entity names
            best_sub = None
            for key, entries in _ENTITY_NAMES_LOWER.items():
                if norm in key or key in norm:
                    best_sub = entries
                    break
            if best_sub:
                for e in best_sub:
                    if e["id"] not in seen_ids:
                        found.append({**e, "match_level": "substring", "confidence": 0.85})
                        seen_ids.add(e["id"])
                continue

        # 1C: Fuzzy (Levenshtein < 3, name len >= 5)
        best_dist, best_entries = 99, []
        for key, entries in _ENTITY_NAMES_LOWER.items():
            if len(key) < 5: continue
            d = _levenshtein(t_lower, key)
            if d <= 2 and d < best_dist:
                best_dist, best_entries = d, entries
        if best_entries and best_dist <= 2:
            for e in best_entries:
                if e["id"] not in seen_ids:
                    found.append({**e, "match_level": "fuzzy", "confidence": round(1.0 - best_dist*0.15, 2)})
                    seen_ids.add(e["id"])
            continue

        # 1D: Vector fallback
        try:
            from src.retrieval.vector_store import get_store
            vs = get_store()
            vec_results = vs.search(token, top_k=1)
            if vec_results and vec_results[0].get("score", 0) >= 0.7:
                r = vec_results[0]
                if r["id"] not in seen_ids:
                    found.append({"id": r["id"], "name": r["name"], "label": r.get("label",""),
                                  "lat": r.get("lat"), "lon": r.get("lon"), "tt": r.get("tt",""),
                                  "match_level": "vector", "confidence": r["score"]})
                    seen_ids.add(r["id"])
                continue
        except Exception:
            pass

        unmatched.append(token)

    # 1E: Filter Holiday entities unless query is about holidays
    q_lower = query.lower()
    has_holiday_intent = any(kw in q_lower for kw in ["holiday","假期","festival","public holiday"])
    if not has_holiday_intent:
        found = [e for e in found if e.get("label","") != "Holiday"]

    # 1F: Ambiguity resolution — MRT > PlanningArea > Bus
    type_priority = {"mrt": 0, "planning_area": 0, "bus": 1}
    found.sort(key=lambda e: type_priority.get(e.get("tt",""), type_priority.get(e.get("label","").lower(), 99)))
    ambiguous = len(set(e["name"].lower() for e in found)) < len(found)

    return {
        "entities": found[:5],
        "unmatched_tokens": unmatched,
        "ambiguous": ambiguous,
    }


# ═══════════════════════════════════════════════════════════
# LAYER 2: INTENT CLASSIFIER
# ═══════════════════════════════════════════════════════════
def classify_intent(query: str, entities: list[dict]) -> dict[str, Any]:
    """Classify user intent. LLM first, fallback to regex."""
    q = query.lower()
    entity_names = [e["name"] for e in entities] if entities else []

    # Try LLM
    try:
        from src.graphrag.llm_client import LLMClient
        llm = LLMClient()
        prompt = f"""Classify this Singapore urban data query into ONE intent word.

Entities found: {entity_names if entity_names else 'none'}

Options:
- COUNT: asking for a number (how many, total, count)
- LIST: asking to enumerate (list, which, what are)
- COMPARE: comparing two things
- PATH: route or connection between stations
- LOCATE: where something is located
- DESCRIBE: property of something (population, price, weather)
- RANK: ordering (most, least, highest, lowest, top)
- UNKNOWN: none of the above

Query: {query}
Intent:"""
        resp = llm.chat("You classify user queries. Output ONE word.", prompt, max_tokens=10, temperature=0, label="intent_classify")
        intent = resp.strip().upper()
        valid = {"COUNT","LIST","COMPARE","PATH","LOCATE","DESCRIBE","RANK","UNKNOWN"}
        if intent in valid:
            return {"intent": intent, "confidence": 0.95, "source": "llm"}
    except Exception as e:
        logger.debug("LLM intent failed: %s, using regex", e)

    # Regex fallback
    patterns = [
        (r'\b(how many|total|count|number of)\b', 'COUNT'),
        (r'\b(list|which|what are|show|enumerate|display)\b', 'LIST'),
        (r'\b(compare|vs|versus|or|and)\b.*\b(compare|vs|versus|or|and)\b', 'COMPARE'),
        (r'\b(from|to|between|path|route|connected|connection|connect)\b', 'PATH'),
        (r'\b(where|located|in which|which area|planning area)\b', 'LOCATE'),
        (r'\b(population|price|cost|weather|rain|temp|humidity|wind)\b', 'DESCRIBE'),
        (r'\b(most|least|top|bottom|highest|lowest|largest|smallest|ranking)\b', 'RANK'),
    ]
    for pat, intent in patterns:
        if re.search(pat, q):
            return {"intent": intent, "confidence": 0.7, "source": "regex"}
    return {"intent": "UNKNOWN", "confidence": 0.3, "source": "regex"}


# ═══════════════════════════════════════════════════════════
# LAYER 3: SLOT FILLER
# ═══════════════════════════════════════════════════════════
INTENT_TEMPLATE_MAP = {
    ("COUNT","station","transport"): ("station_count", {}),
    ("COUNT","cbd","transport"):     ("mrt_count_cbd", {}),
    ("COUNT","circle","transport"):  ("circle_line_stations", {}),
    ("COUNT","bus","transport"):     ("bus_stop_count", {}),
    ("COUNT","population","demo"):   ("total_population", {}),
    ("COUNT","transaction","housing"): ("hdb_total_transactions", {}),
    ("LIST",None,"transport"):       ("mrt_in_any_area", "area"),
    ("LIST",None,"bus"):             ("bus_stops_in_area", "area"),
    ("LIST",None,"line"):            ("lines_at_station", "station"),
    ("PATH",None,None):              ("path_exists", "from_to"),
    ("LOCATE",None,None):            ("station_area_lookup", "station"),
    ("DESCRIBE","population",None):  ("planning_area_population", "area_name"),
    ("DESCRIBE","price",None):       ("hdb_price_town", "town"),
    ("DESCRIBE","holiday",None):     ("next_holiday", {}),
    ("RANK","population","desc"):    ("largest_population", {}),
    ("RANK","population","asc"):     ("smallest_population", {}),
    ("RANK","mrt","desc"):           ("areas_with_most_mrt", {}),
    ("RANK","mrt","asc"):            ("areas_with_least_mrt", {}),
    ("RANK","price","desc"):         ("hdb_highest_prices", {}),
    ("RANK","connections","desc"):   ("stations_most_connections", {}),
    ("RANK","rain","desc"):          ("rainiest_day", {}),
}


def fill_slots(intent: str, entities: list[dict], query: str) -> dict[str, Any] | None:
    """Map intent+entities to Cypher template+params."""
    q = query.lower()
    if not entities and intent not in ("COUNT","RANK"):
        return None

    attr = None
    for kw, at in [("population","population"),("price","price"),("hdb","price"),
                    ("mrt station","mrt"),("mrt","mrt"),("bus","bus"),
                    ("connection","connections"),("rain","rain"),("weather","rain"),
                    ("holiday","holiday"),("temperature","temp")]:
        if kw in q: attr = at; break
    if not attr and intent == "RANK": attr = "mrt"

    domain = None
    for e in entities:
        lb = str(e.get("label","")).lower()
        tt = str(e.get("tt","")).lower()
        if "planningarea" in lb: domain = "demo"
        elif tt == "mrt": domain = "transport"
        elif tt == "bus": domain = "bus"
    if not domain and attr in ("mrt","connections"): domain = "transport"
    if not domain and attr == "population": domain = "demo"

    # Special: check for CBD/Circle keywords
    for kw in ["cbd","downtown"]:
        if kw in q: attr = "cbd"; domain = "transport"; break
    if "circle line" in q: attr = "circle"; domain = "transport"

    key = (intent, attr, domain)
    map_entry = INTENT_TEMPLATE_MAP.get(key)
    if not map_entry:
        # Try without domain
        map_entry = INTENT_TEMPLATE_MAP.get((intent, attr, None))
    if not map_entry:
        # Try without attr
        map_entry = INTENT_TEMPLATE_MAP.get((intent, None, domain))
    if not map_entry:
        return None

    template, param_mode = map_entry
    params = {}

    if param_mode == "area":
        params["area"] = entities[0]["name"]
    elif param_mode == "station":
        params["station"] = entities[0]["name"]
    elif param_mode == "area_name":
        params["area_name"] = entities[0]["name"]
    elif param_mode == "town":
        params["town"] = entities[0]["name"]
    elif param_mode == "from_to":
        if len(entities) >= 2:
            params["from"] = entities[0]["name"]
            params["to"] = entities[1]["name"]

    return {"template": template, "params": params, "intent": intent, "confidence": "HIGH"}


# ═══════════════════════════════════════════════════════════
# LAYER 4: VALIDATOR + EXECUTOR
# ═══════════════════════════════════════════════════════════
def execute_parsed(slots: dict[str, Any]) -> dict[str, Any] | None:
    """Execute a parsed query. Returns formatted result or None if failed."""
    template = slots.get("template")
    params = slots.get("params", {})

    # Validate
    if not template: return None
    for k, v in params.items():
        if v is None or (isinstance(v, str) and not v.strip()):
            logger.warning("Empty param %s, cannot execute %s", k, template)
            return None

    try:
        from src.retrieval.cypher_agent import run_preset
        r = run_preset(template, params if params else None)
        if "error" in r or not r.get("results"):
            return None
        return {
            "status": "success",
            "results": r["results"],
            "count": r["count"],
            "template": template,
            "params": params,
        }
    except Exception as e:
        logger.warning("Cypher execution failed: %s", e)
        return None


# ═══════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════
def parse_and_execute(query: str) -> dict[str, Any] | None:
    """Full pipeline: link entities → classify → fill slots → execute.

    Returns formatted Cypher result dict, or None if pipeline can't serve.
    """
    t0 = time.time()

    # Layer 1
    linked = link_entities(query)
    entities = linked["entities"]
    unmatched = linked["unmatched_tokens"]
    logger.debug("L1: %d entities linked, %d unmatched (%.0fms)",
                 len(entities), len(unmatched), (time.time()-t0)*1000)

    if not entities:
        return None

    # Layer 2
    intent_data = classify_intent(query, entities)
    intent = intent_data["intent"]
    logger.debug("L2: intent=%s source=%s (%.0fms)", intent, intent_data["source"], (time.time()-t0)*1000)

    if intent == "UNKNOWN":
        return None

    # Layer 3
    slots = fill_slots(intent, entities, query)
    if not slots:
        logger.debug("L3: no template for intent=%s", intent)
        return None
    logger.debug("L3: template=%s params=%s (%.0fms)", slots["template"], slots["params"], (time.time()-t0)*1000)

    # Layer 4
    result = execute_parsed(slots)
    if result:
        logger.debug("L4: success (%d rows, %.0fms total)", result["count"], (time.time()-t0)*1000)
        return result

    return None



logger.info("Query parser loaded. Call parse_and_execute(query) for 4-layer pipeline.")
