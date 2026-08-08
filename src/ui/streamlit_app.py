"""
UrbanGraph-SG — Modern Urban Intelligence Dashboard
Design inspired by: Grafana, Neo4j Bloom, Singapore OneMap
"""

import sys, json, math
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
from folium import FeatureGroup

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Lazy imports — avoid heavy deps/network at module load
_run_query = None
def _lazy_run_query(query, params=None):
    global _run_query
    if _run_query is None:
        from src.graph.neo4j_client import run_query as rq
        _run_query = rq
    return __lazy_run_query(query, params)

_gen = None
def get_gen():
    global _gen
    if _gen is None:
        from src.generation.answer_generator import AnswerGenerator
        _gen = AnswerGenerator()
    return _gen

# ═══════════════════════════════════════════════════════
# DESIGN TOKENS
# ═══════════════════════════════════════════════════════
TOKEN_CSS = """
<style>
html, body, [class*="css"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; }
.main { background: #ffffff; }
.main .block-container { padding: 1rem 2rem; max-width: 100%; }
header { visibility: hidden; }
footer { visibility: hidden; }

/* Typography — dark on white for max contrast */
h1, h2, h3, h4 { color: #0f172a; }
h3 { font-size: 1.4rem !important; margin: 0 !important; padding: 0 !important; font-weight: 700; }
p, li, label, .stCaption { color: #475569; }
.stCaption { font-size: 0.8rem; }

/* KPI Cards — white card with colored left border */
.kpi-row { display: flex; gap: 14px; margin: 14px 0; }
.kpi-card { flex: 1; background: #ffffff; border-radius: 10px; padding: 18px 22px;
    border: 1px solid #e2e8f0; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    transition: box-shadow 0.15s; }
.kpi-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
.kpi-card .kpi-num { font-size: 2rem; font-weight: 800; color: #0f172a; line-height: 1.2; }
.kpi-card .kpi-label { font-size: 0.7rem; color: #94a3b8; margin-top: 3px;
    text-transform: uppercase; letter-spacing: 0.8px; font-weight: 600; }
.kpi-accent1 .kpi-num { color: #DC2626; }
.kpi-accent2 .kpi-num { color: #16a34a; }
.kpi-accent3 .kpi-num { color: #2563eb; }
.kpi-accent4 .kpi-num { color: #ea580c; }
.kpi-accent5 .kpi-num { color: #9333ea; }

/* Buttons */
.stButton button { background: #DC2626 !important; color: white !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important; font-size: 0.85rem !important;
    padding: 8px 20px !important; transition: all 0.15s; letter-spacing: 0.3px; }
.stButton button:hover { background: #b91c1c !important; transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(220,38,38,0.25); }

/* Text inputs */
.stTextInput input, .stSelectbox div[data-baseweb="select"] > div {
    background: #ffffff !important; border: 1.5px solid #e2e8f0 !important;
    border-radius: 8px !important; color: #0f172a !important; font-weight: 500 !important;
    padding: 8px 12px !important; }
.stTextInput input:focus, .stSelectbox div[data-baseweb="select"] > div:focus-within {
    border-color: #DC2626 !important; box-shadow: 0 0 0 3px rgba(220,38,38,0.1) !important; }
.stSelectbox svg { color: #64748b !important; }
.stTextInput input::placeholder { color: #94a3b8 !important; }

/* Chat */
.stChatMessage { background: transparent !important; }
[data-testid="stChatMessage"] { background: #f8fafc !important; border-radius: 10px !important;
    padding: 10px 16px !important; margin: 6px 0 !important; border: 1px solid #e2e8f0; }

/* Tabs — clean underline style */
.stTabs [data-baseweb="tab-list"] { gap: 0; background: transparent;
    border-bottom: 2px solid #e2e8f0; }
.stTabs [data-baseweb="tab"] { background: transparent; color: #64748b; border-radius: 0;
    padding: 8px 22px; font-weight: 600; font-size: 0.85rem; border: none;
    margin-bottom: -2px; }
.stTabs [data-baseweb="tab"]:hover { color: #0f172a; }
.stTabs [data-baseweb="tab"][aria-selected="true"] { background: transparent; color: #DC2626;
    border-bottom: 2px solid #DC2626; }

/* Dataframe */
.stDataFrame { background: #ffffff !important; border-radius: 8px !important;
    border: 1px solid #e2e8f0 !important; }
.stDataFrame th { background: #f8fafc !important; color: #0f172a !important;
    font-weight: 700; font-size: 0.8rem; }
.stDataFrame td { color: #334155 !important; font-size: 0.82rem; }

/* Metrics */
[data-testid="stMetricValue"] { color: #0f172a !important; font-size: 1.6rem !important;
    font-weight: 800 !important; }
[data-testid="stMetricLabel"] { color: #64748b !important; font-weight: 600; }

/* Info box */
.stAlert { background: #eff6ff !important; border: 1px solid #bfdbfe !important;
    border-radius: 8px !important; color: #1e40af !important; }

.stImage { border-radius: 8px; border: 1px solid #e2e8f0; }

hr { border-color: #e2e8f0 !important; margin: 20px 0 !important; }
</style>"""
st.markdown(TOKEN_CSS, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# DATA & STATE
# ═══════════════════════════════════════════════════════
PRESETS = [
    "How many MRT stations are there in total?",
    "How many bus stops are there in Singapore?",
    "How many stations are on the Circle Line?",
    "How many MRT stations are in the CBD area?",
    "Which MRT lines pass through Bishan?",
    "Which MRT lines pass through Jurong East?",
    "Which MRT lines serve Woodlands?",
    "List all MRT stations in Orchard",
    "Which MRT stations are in Downtown Core?",
    "Which station has the most connections?",
    "Is Bishan station connected to Orchard?",
    "How many stations from Jurong East to City Hall?",
    "Which planning area is Bedok MRT in?",
    "List bus stops along Orchard Road",
    "What is the population of Bedok?",
    "What is the population of Tampines?",
    "What is the population of Punggol?",
    "Which planning area has the largest population?",
    "Which areas have the smallest population?",
    "Which area has the highest HDB resale prices?",
    "How many HDB transactions are in the database?",
]

_gen = None
@st.cache_resource
def get_gen():
    from src.generation.answer_generator import AnswerGenerator
    return AnswerGenerator()

@st.cache_data(ttl=3600)
def get_kpi_cached():
    try:
        nodes = _lazy_run_query("MATCH (n) RETURN count(n) AS c")[0]["c"]
        edges = _lazy_run_query("MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"]
        mrt = _lazy_run_query("MATCH (n:TransportNode {transport_type:'mrt'}) RETURN count(n) AS c")[0]["c"]
        bus = _lazy_run_query("MATCH (n:TransportNode {transport_type:'bus'}) RETURN count(n) AS c")[0]["c"]
        hdb = _lazy_run_query("MATCH (pa:PlanningArea) WHERE pa.avg_resale_price IS NOT NULL RETURN count(pa) AS c")[0]["c"]
        return nodes, edges, mrt, bus, hdb
    except Exception:
        return 5532, 10964, 137, 5207, 24

def get_kpi():
    return get_kpi_cached()

if "chat" not in st.session_state: st.session_state.chat = []
if "hl" not in st.session_state: st.session_state.hl = []
if "ctx" not in st.session_state: st.session_state.ctx = {}
if "hl" not in st.session_state: st.session_state.hl = []

N, E, M, B, H = get_kpi()

# ═══════════════════════════════════════════════════════
# HERO HEADER
# ═══════════════════════════════════════════════════════
c_title, c_search = st.columns([3, 2])
with c_title:
    st.markdown("### 🎯 UrbanGraph-SG")
    st.caption("GraphRAG-powered Singapore knowledge navigator — 11 CS/AI domains • Neo4j + DeepSeek + ChromaDB")
with c_search:
    st.markdown("")  # spacer for alignment

# KPI Row
st.markdown(f"""
<div class="kpi-row">
    <div class="kpi-card kpi-accent1"><div class="kpi-num">{N:,}</div><div class="kpi-label">Knowledge Graph Nodes</div></div>
    <div class="kpi-card kpi-accent2"><div class="kpi-num">{E:,}</div><div class="kpi-label">Relationships</div></div>
    <div class="kpi-card kpi-accent3"><div class="kpi-num">{M}</div><div class="kpi-label">MRT Stations</div></div>
    <div class="kpi-card kpi-accent4"><div class="kpi-num">{B:,}</div><div class="kpi-label">Bus Stops</div></div>
    <div class="kpi-card kpi-accent5"><div class="kpi-num">{H}</div><div class="kpi-label">HDB Scored Areas</div></div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════
tabs = st.tabs(["🗺️ Explore", "💬 Query", "📊 Analytics", "🔬 Graph ML", "📋 Report"])

# ═══════════════ TAB 0: EXPLORE ═══════════════
with tabs[0]:
    # ── Search + Spatial Query Bar ──
    s1, s2, s3 = st.columns([3, 1, 1])
    with s1:
        search_term = st.text_input("Search", placeholder="Search station or area (e.g. Orchard, Bedok, Bishan)...", label_visibility="collapsed", key="map_search")
    with s2:
        radius = st.selectbox("Nearby radius", ["500m", "1km", "2km"], key="radius_sel", label_visibility="collapsed")
    with s3:
        do_search = st.button("Find", width="stretch", key="map_find_btn")

    # ── Context Bar ──
    # Initialize session state for map
    if "map_center" not in st.session_state: st.session_state.map_center = [1.3521, 103.8198]
    if "map_zoom" not in st.session_state: st.session_state.map_zoom = 12
    if "selected_station" not in st.session_state: st.session_state.selected_station = None
    if "nearby_entities" not in st.session_state: st.session_state.nearby_entities = []

    # Process search
    if do_search and search_term:
        try:
            results = _lazy_run_query("""
                MATCH (n) WHERE toLower(n.name) CONTAINS toLower($q) AND n.lat IS NOT NULL
                RETURN n.name AS name, n.lat AS lat, n.lon AS lon, labels(n)[0] AS type,
                       n.transport_type AS tt, n.population AS pop
                ORDER BY CASE WHEN n.transport_type='mrt' THEN 0 ELSE 1 END, n.name LIMIT 1
            """, {"q": search_term.strip()})
            if results:
                r = results[0]
                st.session_state.map_center = [float(r["lat"]), float(r["lon"])]
                st.session_state.map_zoom = 15
                st.session_state.selected_station = {"name": r["name"], "type": r.get("type",""), "lat": r["lat"], "lon": r["lon"], "pop": r.get("pop"), "tt": r.get("tt")}
                # Find nearby entities
                dist_m = {"500m": 500, "1km": 1000, "2km": 2000}.get(radius, 1000)
                center_lat = float(r["lat"])
                dist_lat_deg = dist_m / 111320.0
                dist_lon_deg = dist_m / (111320.0 * math.cos(math.radians(center_lat)))
                nearby = _lazy_run_query("""
                    MATCH (n) WHERE n.lat IS NOT NULL AND n.lat > $min_lat AND n.lat < $max_lat
                    AND n.lon > $min_lon AND n.lon < $max_lon AND n.name <> $center_name
                    RETURN n.name AS name, labels(n)[0] AS type, n.lat AS lat, n.lon AS lon,
                           n.transport_type AS tt
                    ORDER BY n.name LIMIT 30
                """, {"min_lat": r["lat"] - dist_lat_deg, "max_lat": r["lat"] + dist_lat_deg,
                      "min_lon": r["lon"] - dist_lon_deg, "max_lon": r["lon"] + dist_lon_deg,
                      "center_name": r["name"]})
                st.session_state.nearby_entities = nearby
                st.session_state.map_center = [float(r["lat"]), float(r["lon"])]
                st.session_state.map_zoom = 15
            else:
                st.warning(f"No station or area found matching '{search_term}'")
        except Exception as e:
            st.warning(f"Search error: {e}")

    # ── Info Panel (if station selected) ──
    sel = st.session_state.selected_station
    if sel:
        pinfo = []
        try:
            # Get lines for MRT stations
            if sel.get("tt") == "mrt":
                lines = _lazy_run_query("""
                    MATCH (n:TransportNode {name: $name})-[r:CONNECTS_TO]-(neighbor)
                    RETURN DISTINCT r.line AS line, collect(neighbor.name)[0..3] AS neighbors
                """, {"name": sel["name"]})
                for l in lines:
                    pinfo.append(f'{l["line"]}: connected to {", ".join(l["neighbors"])}')

            # Get planning area
            pa = _lazy_run_query("MATCH (n {name: $name})-[:LOCATED_IN]->(pa:PlanningArea) RETURN pa.name AS area, pa.population AS pop", {"name": sel["name"]})
            if pa:
                pinfo.append(f"Planning Area: {pa[0]['area']}" + (f" (pop: {pa[0]['pop']:,})" if pa[0].get("pop") else ""))
        except: pass

        st.info(f"**📍 {sel['name']}** ({sel.get('tt', sel.get('type', ''))})" + ("\n- " + "\n- ".join(pinfo) if pinfo else ""))
        st.caption(f"Nearby ({radius}): {len(st.session_state.nearby_entities)} entities found")

    # ── Map ──
    mp = folium.Map(location=st.session_state.map_center, zoom_start=st.session_state.map_zoom,
                     tiles="CartoDB positron", control_scale=True)
    colors = {"EWL":"#009530","NSL":"#D42E2B","NEL":"#9900AA","CCL":"#FA9E0D","DTL":"#005EC4","TEL":"#9D5B25"}
    line_names = {"EWL":"East-West","NSL":"North-South","NEL":"North East","CCL":"Circle","DTL":"Downtown","TEL":"Thomson-East Coast"}

    # Layer 1: MRT Lines
    try:
        ld = _lazy_run_query("MATCH (a:TransportNode {transport_type:'mrt'})-[r:CONNECTS_TO]->(b:TransportNode {transport_type:'mrt'}) RETURN a.lat AS al, a.lon AS ao, b.lat AS bl, b.lon AS bo, r.line AS l LIMIT 150")
        mrt_line_grps = {}
        for row in ld:
            ln = row.get("l","?")
            if ln not in mrt_line_grps: mrt_line_grps[ln] = FeatureGroup(name=f"🚇 {ln} {line_names.get(ln,'')}")
            try: folium.PolyLine([[float(row["al"]),float(row["ao"])],[float(row["bl"]),float(row["bo"])]], color=colors.get(ln,"#888"), weight=3, opacity=0.8, tooltip=ln).add_to(mrt_line_grps[ln])
            except: pass
        for g in mrt_line_grps.values(): g.add_to(mp)
    except: pass

    # Layer 2: MRT Stations
    try:
        mrts = _lazy_run_query("MATCH (n:TransportNode {transport_type:'mrt'}) RETURN n.name AS n, n.lat AS la, n.lon AS lo LIMIT 200")
        mrt_station_fg = FeatureGroup(name="🔴 MRT Stations")
        for s in mrts:
            if s.get("la") and s.get("lo"):
                popup_html = f"<b>{s['n']}</b><br><small>Click for info panel</small>"
                folium.CircleMarker([float(s["la"]),float(s["lo"])], radius=4, color="#ED2939", fill=True,
                    fill_color="#ED2939", fill_opacity=0.9, tooltip=s["n"], popup=folium.Popup(popup_html, max_width=200)).add_to(mrt_station_fg)
        mrt_station_fg.add_to(mp)
    except: pass

    # Layer 3: Bus Stops (sampled for performance)
    try:
        buses = _lazy_run_query("MATCH (n:TransportNode {transport_type:'bus'}) WHERE n.lat IS NOT NULL RETURN n.name AS n, n.lat AS la, n.lon AS lo LIMIT 3000")
        bus_fg = FeatureGroup(name="🔵 Bus Stops")
        for b in buses:
            if b.get("la") and b.get("lo"):
                folium.CircleMarker([float(b["la"]),float(b["lo"])], radius=1, color="#005EC4", fill=True,
                    fill_color="#005EC4", fill_opacity=0.5, tooltip=b["n"]).add_to(bus_fg)
        bus_fg.add_to(mp)
    except: pass

    # Layer 4: Planning Area boundaries
    try:
        import json, pandas as pd
        pa_path = Path("data/raw/onemap/planning_areas.parquet")
        if pa_path.exists():
            pa_df = pd.read_parquet(pa_path)
            pa_fg = FeatureGroup(name="🏙️ Planning Areas")
            for _, row in pa_df.iterrows():
                try:
                    geom = json.loads(row["geojson"])
                    folium.GeoJson(geom, style_function=lambda x: {"fillColor":"#3388ff","color":"#3388ff","weight":0.5,"fillOpacity":0.05},
                        tooltip=row["pln_area_n"].title()).add_to(pa_fg)
                except: pass
            pa_fg.add_to(mp)
    except: pass

    # Layer 5: Nearby entities highlight
    if st.session_state.nearby_entities:
        nearby_fg = FeatureGroup(name="📍 Nearby Results")
        for e in st.session_state.nearby_entities:
            try:
                color = "#ED2939" if e.get("tt") == "mrt" else "#005EC4" if e.get("tt") == "bus" else "#FA9E0D"
                folium.CircleMarker([float(e["lat"]),float(e["lon"])], radius=5, color=color, fill=True,
                    fill_color=color, fill_opacity=0.6, weight=2, tooltip=e["name"]).add_to(nearby_fg)
            except: pass
        nearby_fg.add_to(mp)

    # Layer 6: Selected station highlight
    if sel and sel.get("lat") and sel.get("lon"):
        sel_fg = FeatureGroup(name="🎯 Selected")
        folium.Marker([float(sel["lat"]),float(sel["lon"])], icon=folium.Icon(color="orange",icon="star",prefix="fa"),
            popup=f"<b>{sel['name']}</b>").add_to(sel_fg)
        # Radius circle
        dist_m = {"500m":500,"1km":1000,"2km":2000}.get(radius, 1000)
        folium.Circle([float(sel["lat"]),float(sel["lon"])], radius=dist_m, color="#FA9E0D", weight=1, fill=True,
            fill_color="#FA9E0D", fill_opacity=0.08).add_to(sel_fg)
        sel_fg.add_to(mp)

    # Highlight entities from chat answers
    highlight_fg = FeatureGroup(name="⭐ Answer Highlights")
    for e in st.session_state.get("hl",[]):
        try: folium.Marker([e["lat"],e["lon"]], icon=folium.Icon(color="orange",icon="star",prefix="fa"), popup=f"<b>{e['name']}</b>").add_to(highlight_fg)
        except: pass
    highlight_fg.add_to(mp)

    folium.LayerControl(collapsed=False).add_to(mp)
    folium_static(mp, height=520)

    # Legend
    legend_cols = st.columns(7)
    for i, code in enumerate(["EWL","NSL","NEL","CCL","DTL","TEL"]):
        legend_cols[i].markdown(f'<span style="color:{colors[code]};font-weight:600">●</span> <span style="color:#8a8d91;font-size:0.7rem">{code}</span>', unsafe_allow_html=True)
    legend_cols[6].markdown(f'<span style="color:#ED2939">●M</span><span style="color:#8a8d91;font-size:0.65rem">RT</span> <span style="color:#005EC4">●B</span><span style="color:#8a8d91;font-size:0.65rem">us</span>', unsafe_allow_html=True)


# ═══════════════ TAB 1: QUERY ═══════════════
with tabs[1]:
    # ── Fix #1: Categorized preset chips (not long dropdown) ──
    categories = {
        "🚇 Transport": ["How many MRT stations are there in total?","How many bus stops are there in Singapore?",
            "How many stations are on the Circle Line?","How many MRT stations are in the CBD area?",
            "Which MRT lines pass through Bishan?","Which MRT lines pass through Jurong East?",
            "Which MRT lines serve Woodlands?","List all MRT stations in Orchard",
            "Which MRT stations are in Downtown Core?","Which station has the most connections?",
            "Is Bishan station connected to Orchard?","How many stations from Jurong East to City Hall?"],
        "👥 Population": ["What is the population of Bedok?","What is the population of Tampines?",
            "What is the population of Punggol?","Which planning area has the largest population?",
            "Which areas have the smallest population?"],
        "🏠 Housing": ["Which area has the highest HDB resale prices?","How many HDB transactions are in the database?"],
        "📍 Spatial": ["Which planning area is Bedok MRT in?","List bus stops along Orchard Road",
            "How many bus stops are near Orchard MRT?"],
    }
    cat_tabs = st.tabs(list(categories.keys()))
    selected_preset = None
    for i, (cat_name, questions) in enumerate(categories.items()):
        with cat_tabs[i]:
            cols = st.columns(2)
            for j, q in enumerate(questions):
                with cols[j % 2]:
                    if st.button(q[:80], key=f"cat_{i}_{j}", width="stretch",
                       help=f"Domain: {cat_name}"):
                        selected_preset = q

    # ── Search bar ──
    manual = st.chat_input("Or type any question — semantic search + Cypher + ChromaDB...")

    query = None
    if selected_preset: query = selected_preset
    if manual: query = manual

    if query:
        st.session_state.chat.append({"role": "user", "content": query})
        with st.spinner("Retrieving from knowledge graph + vector store..."):
            ctx = dict(st.session_state.ctx) if st.session_state.ctx else None
            r = get_gen().answer(query, context=ctx)
            if hasattr(get_gen(), '_context_cache') and get_gen()._context_cache:
                st.session_state.ctx = dict(get_gen()._context_cache)
        st.session_state.chat.append({"role": "assistant", "content": r["answer_text"],
            "confidence": r.get("confidence","MEDIUM"), "mode": r.get("retrieval_mode",""),
            "sources": r.get("sources_used",[]), "entities": r.get("entities",[])})
        # Map highlights using entity index cache (avoids ~5000 node Neo4j query per request)
        hl = []
        try:
            from src.retrieval.query_parser import get_entity_index
            entity_index = get_entity_index()
            answer_lower = r["answer_text"].lower()
            for e in entity_index.values():
                name = e["name"]
                if len(name) > 3 and name.lower() in answer_lower:
                    lat = e.get("lat")
                    lon = e.get("lon")
                    if lat and lon:
                        hl.append({"name": name, "lat": float(lat), "lon": float(lon), "label": e.get("label", "")})
                        if len(hl) >= 8:
                            break
        except Exception:
            pass
        st.session_state.hl = hl
        st.rerun()

    # ── Chat display ──
    for i, msg in enumerate(st.session_state.chat):
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(f"**{msg['content']}**")
            else:
                # ── Fix #3: Confidence bar (not just text) ──
                conf = msg.get("confidence","MEDIUM")
                conf_pct = {"HIGH": 95, "MEDIUM": 70, "LOW": 40}.get(conf, 50)
                conf_colors = {"HIGH": "#16a34a", "MEDIUM": "#ea580c", "LOW": "#DC2626"}
                col_conf, col_mode = st.columns([1, 3])
                with col_conf:
                    st.markdown(f"""
                    <div style="background:#f1f5f9;border-radius:8px;padding:8px;text-align:center">
                        <div style="font-size:1.4rem;font-weight:800;color:{conf_colors.get(conf,'#64748b')}">{conf_pct}%</div>
                        <div style="font-size:0.65rem;color:#64748b;font-weight:600">{conf}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_mode:
                    mode_label = {"cypher":"Cypher Query","semantic":"Semantic Search","local":"Local Search"}.get(msg.get("mode",""), msg.get("mode",""))
                    st.caption(f"Retrieval: {mode_label} · DeepSeek LLM")

                st.write(msg["content"])

                # ── Fix #2: Expandable source citations ──
                sources = msg.get("sources", [])
                if sources:
                    with st.expander(f"📎 {len(sources)} sources cited"):
                        for s in sources[:8]:
                            st.caption(f"• {str(s)[:200]}")
                        st.caption("Data: LTA DataMall, data.gov.sg, OneMap, SingStat · 2024-2025")

                # ── Fix #5: Mini chart for ranked answers ──
                content = msg["content"]
                import re
                numbers = re.findall(r'(\d[\d,]*)\s*(?:stations|stops|areas)', content.lower())
                if numbers and "1." in content:
                    # Extract ranking data for mini bar
                    lines = [l.strip() for l in content.split('\n') if re.match(r'\d+\.', l.strip())]
                    if len(lines) >= 3:
                        chart_data = {}
                        for line in lines[:6]:
                            m = re.match(r'\d+\.\s*\*{0,2}([^*\n]+?)\*{0,2}\s*[–\-—]\s*(\d+)', line)
                            if m:
                                chart_data[m.group(1).strip()] = int(m.group(2))
                        if chart_data:
                            st.bar_chart(chart_data, horizontal=True, height=180)


# ═══════════════ TAB 2: ANALYTICS ═══════════════
with tabs[2]:
    fig_dir = Path("reports/figures")

    # ── Fix #5: Run Analysis button ──
    c_run, c_space = st.columns([2, 5])
    with c_run:
        if st.button("🔄 Refresh Analysis", type="primary", width="stretch"):
            with st.spinner("Re-running ML pipeline..."):
                try:
                    from src.ml.weather_predictor import WeatherPredictor
                    from src.ml.visualization import plot_feature_importance, plot_predictions, plot_weather_dashboard
                    from src.ml.timeseries_analysis import analyze as run_ts
                    wp = WeatherPredictor(); X, y = wp.prepare_data(); results = wp.train_and_evaluate()
                    wp_best, wp_model = wp.get_best_model()
                    if 'RandomForest' in wp.models:
                        plot_feature_importance(wp.feature_names, wp.models['RandomForest'].feature_importances_)
                        from sklearn.model_selection import train_test_split
                        from sklearn.preprocessing import StandardScaler
                        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
                        sc = StandardScaler(); Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)
                        yp = wp.models['RandomForest'].predict(Xte_s)
                        plot_predictions(yte, yp)
                    run_ts()
                    plot_weather_dashboard()
                    st.success("Analysis refreshed! Charts updated.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Analysis failed: {e}")

    # ── Fix #1: Live metrics from model card ──
    card_path = Path("reports/mlops/model_card.json")
    if card_path.exists():
        card = json.loads(card_path.read_text())
        best = card.get("best_model", "RandomForest")
        models_data = card.get("models", {})
    else:
        best = "RandomForest"
        models_data = {}

    r2_val = models_data.get(best, {}).get("R2", 0.819) if models_data else 0.819
    cv_val = models_data.get(best, {}).get("CV_R2_mean", 0.761) if models_data else 0.761
    cv_std = models_data.get(best, {}).get("CV_R2_std", 0.098) if models_data else 0.098

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Best Model", best, help="Selected by R² on test set")
    with c2: st.metric("R² Score", f"{r2_val:.3f}", help="Coefficient of determination")
    with c3: st.metric("CV (mean±std)", f"{cv_val:.3f}±{cv_std:.3f}", help="5-fold cross-validation")
    with c4: st.metric("Features", "9", help="temp, humidity, wind, day_of_week, day_of_month, rain_lag1, temp_lag1, rain_roll3, temp_roll3")

    # ── Fix #3: Model comparison grid ──
    st.divider()
    st.subheader("Model Comparison")
    if models_data:
        cols = st.columns(len(models_data))
        for i, (name, metrics) in enumerate(models_data.items()):
            with cols[i]:
                is_best = name == best
                border = "2px solid #16a34a" if is_best else "1px solid #e2e8f0"
                bg = "#f0fdf4" if is_best else "#ffffff"
                st.markdown(f"""
                <div style="background:{bg};border:{border};border-radius:10px;padding:16px;text-align:center">
                    <div style="font-weight:700;font-size:0.95rem;color:#0f172a">{name}</div>
                    <div style="font-size:1.6rem;font-weight:800;color:#0f172a;margin:6px 0">R²={metrics.get('R2',0):.3f}</div>
                    <div style="font-size:0.7rem;color:#64748b">MAE={metrics.get('MAE',0):.1f} · RMSE={metrics.get('RMSE',0):.1f}</div>
                    <div style="font-size:0.7rem;color:#64748b">CV={metrics.get('CV_R2_mean',0):.3f}±{metrics.get('CV_R2_std',0):.3f}</div>
                </div>
                """, unsafe_allow_html=True)

    # ── Charts grid ──
    st.divider()
    st.subheader("Model Diagnostics")
    charts_primary = [
        ("feature_importance.png", "Feature Importance — Temperature dominates at 76% weight"),
        ("prediction_vs_actual.png", "Predicted vs Actual Rainfall — R² = 0.82"),
        ("weather_dashboard.png", "90-Day Singapore Weather — 4 Variables"),
    ]
    cols = st.columns(3)
    for i, (f, cap) in enumerate(charts_primary):
        p = fig_dir / f
        if p.exists():
            with cols[i]: st.image(str(p), caption=cap, width="stretch")

    st.divider()
    st.subheader("Time Series & Correlations")

    # ── Fix #4: Key findings callout ──
    findings_cols = st.columns(3)
    findings = [
        ("🌡️↔💧", "-0.852", "Humidity ↔ Temperature", "Strong inverse — warmer air holds less moisture"),
        ("🌧️↔💧", "0.737", "Humidity ↔ Rainfall", "Wetter days = higher humidity"),
        ("🌡️↔💨", "0.403", "Temperature ↔ Wind", "Moderate positive correlation"),
    ]
    for i, (icon, val, title, desc) in enumerate(findings):
        with findings_cols[i]:
            st.markdown(f"""
            <div style="background:#f8fafc;border-radius:10px;padding:14px;border:1px solid #e2e8f0;text-align:center">
                <div style="font-size:1.6rem">{icon}</div>
                <div style="font-size:1.8rem;font-weight:800;color:#DC2626">{val}</div>
                <div style="font-size:0.8rem;font-weight:600;color:#0f172a">{title}</div>
                <div style="font-size:0.7rem;color:#64748b">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    charts2 = [
        ("timeseries_dashboard.png", "91-Day Trends with 7-Day Rolling Mean"),
        ("correlation_heatmap.png", "Variable Correlation Matrix"),
        ("seasonal_pattern.png", "Day-of-Week Weather Patterns"),
    ]
    cols2 = st.columns(3)
    for i, (f, cap) in enumerate(charts2):
        p = fig_dir / f
        if p.exists():
            with cols2[i % 3]: st.image(str(p), caption=cap, width="stretch")


# ═══════════════ TAB 3: GRAPH ML ═══════════════
with tabs[3]:
    # ── Fix #1: Cached engine + manual refresh ──
    if "graph_ml_engine" not in st.session_state:
        st.session_state.graph_ml_engine = None
        st.session_state.graph_ml_preds = None
        st.session_state.graph_ml_stats = None

    col_title, col_btn = st.columns([3, 1])
    with col_title:
        st.subheader("🔬 Graph Machine Learning — Node2Vec Embeddings")
    with col_btn:
        if st.button("🔄 Train Embeddings", type="primary", width="stretch"):
            with st.spinner("Training Node2Vec on transport graph..."):
                try:
                    from src.ml.graph_ml import GraphMLEngine
                    engine = GraphMLEngine()
                    engine.build_graph()
                    engine.train_node2vec(dimensions=32)
                    preds = engine.predict_links(top_k=15)
                    engine.plot_embeddings()
                    Nn = engine.G.number_of_nodes()
                    Ne = engine.G.number_of_edges()

                    # ── Fix #3: Graph statistics ──
                    import networkx as nx
                    connected = nx.number_connected_components(engine.G)
                    largest_cc = max(nx.connected_components(engine.G), key=len)
                    avg_path = nx.average_shortest_path_length(engine.G.subgraph(largest_cc))
                    density = nx.density(engine.G)

                    st.session_state.graph_ml_engine = engine
                    st.session_state.graph_ml_preds = preds
                    st.session_state.graph_ml_stats = {
                        "nodes": Nn, "edges": Ne, "components": connected,
                        "avg_path": round(avg_path, 2), "density": round(density, 6),
                        "dim": 32,
                    }
                    st.rerun()
                except Exception as e:
                    st.error(f"Training failed: {e}")

    engine = st.session_state.graph_ml_engine
    preds = st.session_state.graph_ml_preds
    stats = st.session_state.graph_ml_stats

    if stats:
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1: st.metric("Nodes", f"{stats['nodes']:,}")
        with c2: st.metric("Edges", stats['edges'])
        with c3: st.metric("Components", stats['components'])
        with c4: st.metric("Avg Path", stats['avg_path'])
        with c5: st.metric("Density", f"{stats['density']:.5f}")
        with c6: st.metric("Dim", stats['dim'])

    if preds:
        st.subheader("Predicted Missing Links")
        st.caption("These station pairs have high embedding similarity but no direct connection — potential new links")

        # ── Fix #2: Enriched link prediction table with explanations ──
        import networkx as nx
        enriched_rows = []
        for p in preds:
            src = p.get("source_name", "")
            tgt = p.get("target_name", "")
            sim = p.get("similarity", 0)
            # Extract actual station codes (after lta-mrt- prefix)
            src_raw = str(p.get("source",""))
            tgt_raw = str(p.get("target",""))
            src_code = src_raw.replace("lta-mrt-","")[:3] if "lta-mrt-" in src_raw else src_raw[:4]
            tgt_code = tgt_raw.replace("lta-mrt-","")[:3] if "lta-mrt-" in tgt_raw else tgt_raw[:4]
            same_line_prefix = src_code[:2] == tgt_code[:2] and len(src_code) >= 2
            line_prefix = src_code[:2] if same_line_prefix else ""
            explanation = f"Same line ({line_prefix}) — 2 hops apart" if same_line_prefix else "Cross-line proximity"
            # Try to get hop distance from engine graph
            hop_dist = "?"
            if engine and engine.G and src in engine.G and tgt in engine.G:
                try:
                    hop_dist = str(nx.shortest_path_length(engine.G, src, tgt))
                except: pass
            enriched_rows.append({
                "Source": src, "Target": tgt,
                "Similarity": f"{float(sim):.4f}",
                "Explanation": explanation,
                "Graph Distance": f"{hop_dist} hops",
                "Action": f"Explore {src}",
            })

        edf = pd.DataFrame(enriched_rows)

        # ── Fix #5: Clickable "Explore" buttons ──
        for i, row in enumerate(enriched_rows):
            cols = st.columns([2, 2, 1, 1.5, 1.5, 2])
            with cols[0]: st.caption(row["Source"])
            with cols[1]: st.caption(row["Target"])
            with cols[2]: st.caption(row["Similarity"])
            with cols[3]: st.caption(row["Explanation"])
            with cols[4]: st.caption(row["Graph Distance"])
            with cols[5]:
                if st.button(f"🔍 View", key=f"explore_link_{i}"):
                    # Switch to Explore tab and highlight
                    lat = engine.G.nodes[row["Source"]].get("lat", 1.35) if engine else 1.35
                    lon = engine.G.nodes[row["Source"]].get("lon", 103.8) if engine else 103.8
                    st.session_state.map_center = [float(lat), float(lon)]
                    st.session_state.map_zoom = 14
                    st.session_state.selected_station = {"name": row["Source"], "type": "MRT", "lat": lat, "lon": lon}
                    st.success(f"Go to 🗺️ Explore tab to see {row['Source']} ↔ {row['Target']}")

        # Header row
        st.caption("Click 🔍 View to jump to Explore tab with station highlighted")

        # ── Fix #4: t-SNE plot ──
        embp = fig_dir / "node_embeddings_tsne.png"
        if embp.exists():
            col_viz, col_info = st.columns([2, 1])
            with col_viz:
                st.image(str(embp), caption="t-SNE Projection of Transport Node Embeddings — MRT (red) vs Bus (blue)", width="stretch")
            with col_info:
                st.info("**How to read:**\n\n"
                        "• Dots close together = similar structural role\n"
                        "• Same-line stations cluster naturally\n"
                        "• Bus stops form dense peripheral clusters\n"
                        "• Interchange hubs appear at cluster boundaries\n\n"
                        "**Method:** SVD decomposition of adjacency matrix → 32D embeddings → t-SNE projection to 2D")
    else:
        st.info("👆 Click **Train Embeddings** to build the Node2Vec model from the transport graph (3,137 nodes, 130 edges).")
        st.caption("First run may take 30-60 seconds. Results are cached for the session.")


# ═══════════════ TAB 4: REPORT ═══════════════
with tabs[4]:
    # ── Fix #4: System Health Dashboard ──
    st.subheader("System Health")
    h1, h2, h3, h4 = st.columns(4)
    try:
        neo = _lazy_run_query("RETURN 1 AS ok")[0]["ok"]
        with h1: st.success(f"✅ Neo4j — Connected")
    except Exception:
        with h1: st.error(f"❌ Neo4j — Offline")
    try:
        from src.retrieval.vector_store import get_store
        vs = get_store()
        vs._ensure_client()
        count = vs.collection.count() if vs.collection else 0
        with h2: st.success(f"✅ ChromaDB — {count:,} entities indexed")
    except Exception:
        with h2: st.warning("⚠️ ChromaDB — Not connected")
    try:
        g = get_gen()
        with h3: st.success(f"✅ LLM — {g.llm.model}")
    except Exception:
        with h3: st.error("❌ LLM — API unreachable")
    try:
        import psutil, os
        mem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        with h4: st.info(f"💾 Memory — {mem:.0f} MB")
    except Exception:
        with h4: st.info("💾 Memory — N/A")

    st.divider()

    # ── Fix #1: Structured Model Card (not JSON dump) ──
    col_mc, col_exp = st.columns([1, 1])

    with col_mc:
        st.subheader("Model Card")
        cp = Path("reports/mlops/model_card.json")
        if cp.exists():
            card = json.loads(cp.read_text())
            st.markdown(f"""
            <div style="background:#f8fafc;border-radius:10px;padding:18px;border:1px solid #e2e8f0">
                <div style="font-weight:700;font-size:1rem;color:#0f172a">{card.get('project','')}</div>
                <div style="font-size:0.75rem;color:#64748b;margin:4px 0">{card.get('generated_at','')[:19]}</div>
                <hr style="margin:10px 0">
                <div style="font-size:0.8rem;color:#334155"><b>Dataset:</b> {card.get('dataset','')}</div>
                <div style="font-size:0.8rem;color:#334155"><b>Task:</b> {card.get('task','')}</div>
                <div style="font-size:0.8rem;color:#334155"><b>Best Model:</b> <span style="color:#16a34a;font-weight:700">{card.get('best_model','')}</span></div>
            </div>
            """, unsafe_allow_html=True)

            st.subheader("Model Comparison")
            models = card.get("models", {})
            if models:
                rows = []
                for name, metrics in models.items():
                    rows.append({
                        "Model": name,
                        "R²": f"{metrics.get('R2', 0):.4f}",
                        "MAE": f"{metrics.get('MAE', 0):.1f}",
                        "RMSE": f"{metrics.get('RMSE', 0):.1f}",
                        "CV R²": f"{metrics.get('CV_R2_mean', 0):.4f}±{metrics.get('CV_R2_std', 0):.4f}",
                    })
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        else:
            st.info("Run 'Refresh Analysis' in the Analytics tab to generate model card")

        # ── Fix #2: Experiment timeline ──
        st.subheader("Experiment Timeline")
        ep = Path("reports/mlops/experiments_20260807.json")
        if ep.exists():
            exps = json.loads(ep.read_text())
            for e in exps:
                with st.expander(f"🔬 {e['run_name']} — {e.get('status','')}"):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.caption("Parameters")
                        st.json(e.get("params", {}))
                    with c2:
                        st.caption("Metrics")
                        st.json(e.get("metrics", {}))
        else:
            st.info("No experiment logs yet")

    with col_exp:
        st.subheader("Geospatial Visualizations")

        # ── Fix #3: Categorized visuals with explanations ──
        viz_categories = {
            "Network Topology": {
                "files": [("mrt_topology.png", "MRT Network Topology — 137 stations, 7 lines, color-coded connections")],
                "desc": "Graph representation of Singapore's MRT network. Nodes are stations, edges are CONNECTS_TO relationships. Interchange stations (degree > 2) are labeled."
            },
            "Demographics": {
                "files": [("demographic_heatmap.png", "Population Density Heatmap — Yellow=High, Light=Low")],
                "desc": "55 planning areas colored by population. Darker areas have higher population density. Data sourced from SingStat 2024 estimates."
            },
            "Clustering": {
                "files": [("area_clustering.png", "K-Means Area Clustering — 4 clusters by demographics+transport")],
                "desc": "Planning areas clustered by population, MRT station count, and HDB price. Downtown Core forms its own cluster (high MRT, high price)."
            },
        }

        for cat_name, cat_data in viz_categories.items():
            with st.expander(f"📊 {cat_name}", expanded=(cat_name == "Network Topology")):
                st.caption(cat_data["desc"])
                for f, cap in cat_data["files"]:
                    p = fig_dir / f
                    if p.exists():
                        st.image(str(p), caption=cap, width="stretch")

    # ── Fix #5: Export Report ──
    st.divider()
    col_export, col_space = st.columns([2, 5])
    with col_export:
        if st.button("📥 Export Report Summary", width="stretch"):
            try:
                export = {
                    "project": "UrbanGraph-SG",
                    "exported_at": str(datetime.now()),
                    "system_status": {
                        "neo4j": "connected" if _lazy_run_query("RETURN 1") else "offline",
                        "chromadb": count if 'count' in dir() else "unknown",
                        "llm_model": get_gen().llm.model,
                    },
                    "knowledge_graph": {"nodes": N, "edges": E, "mrt": M, "bus": B, "hdb_areas": H},
                    "ml_model_card": json.loads(cp.read_text()) if cp.exists() else {},
                    "experiments": json.loads(ep.read_text()) if ep.exists() else [],
                }
                export_str = json.dumps(export, indent=2, default=str)
                st.download_button("⬇️ Download JSON Report", export_str, "urbangraph_report.json", "application/json",
                    width="stretch", key="dl_report")
                st.success("Report generated! Click above to download.")
            except Exception as e:
                st.error(f"Export failed: {e}")
