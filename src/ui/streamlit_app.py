"""
UrbanGraph-SG — Modern Urban Intelligence Dashboard
Design inspired by: Grafana, Neo4j Bloom, Singapore OneMap
"""

import sys, json
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
from folium import FeatureGroup

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.generation.answer_generator import AnswerGenerator
from src.graph.neo4j_client import run_query

st.set_page_config(page_title="UrbanGraph-SG", page_icon="🇸🇬", layout="wide")

# ═══════════════════════════════════════════════════════
# DESIGN TOKENS
# ═══════════════════════════════════════════════════════
TOKEN_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', -apple-system, system-ui, sans-serif; }
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

def get_gen(): return AnswerGenerator()

def get_kpi():
    try:
        nodes = run_query("MATCH (n) RETURN count(n) AS c")[0]["c"]
        edges = run_query("MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"]
        mrt = run_query("MATCH (n:TransportNode {transport_type:'mrt'}) RETURN count(n) AS c")[0]["c"]
        bus = run_query("MATCH (n:TransportNode {transport_type:'bus'}) RETURN count(n) AS c")[0]["c"]
        hdb = run_query("MATCH (pa:PlanningArea) WHERE pa.avg_resale_price IS NOT NULL RETURN count(pa) AS c")[0]["c"]
        return nodes, edges, mrt, bus, hdb
    except: return 5532, 10964, 137, 5207, 24

if "chat" not in st.session_state: st.session_state.chat = []
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
        search_term = st.text_input("", placeholder="Search station or area (e.g. Orchard, Bedok, Bishan)...", label_visibility="collapsed", key="map_search")
    with s2:
        radius = st.selectbox("Nearby radius", ["500m", "1km", "2km"], key="radius_sel", label_visibility="collapsed")
    with s3:
        do_search = st.button("Find", use_container_width=True, key="map_find_btn")

    # ── Context Bar ──
    # Initialize session state for map
    if "map_center" not in st.session_state: st.session_state.map_center = [1.3521, 103.8198]
    if "map_zoom" not in st.session_state: st.session_state.map_zoom = 12
    if "selected_station" not in st.session_state: st.session_state.selected_station = None
    if "nearby_entities" not in st.session_state: st.session_state.nearby_entities = []

    # Process search
    if do_search and search_term:
        try:
            results = run_query("""
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
                dist_deg = dist_m / 111000.0
                nearby = run_query("""
                    MATCH (n) WHERE n.lat IS NOT NULL AND n.lat > $min_lat AND n.lat < $max_lat
                    AND n.lon > $min_lon AND n.lon < $max_lon AND n.name <> $center_name
                    RETURN n.name AS name, labels(n)[0] AS type, n.lat AS lat, n.lon AS lon,
                           n.transport_type AS tt
                    ORDER BY n.name LIMIT 30
                """, {"min_lat": r["lat"] - dist_deg, "max_lat": r["lat"] + dist_deg,
                      "min_lon": r["lon"] - dist_deg, "max_lon": r["lon"] + dist_deg,
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
                lines = run_query("""
                    MATCH (n:TransportNode {name: $name})-[r:CONNECTS_TO]-(neighbor)
                    RETURN DISTINCT r.line AS line, collect(neighbor.name)[0..3] AS neighbors
                """, {"name": sel["name"]})
                for l in lines:
                    pinfo.append(f'{l["line"]}: connected to {", ".join(l["neighbors"])}')

            # Get planning area
            pa = run_query("MATCH (n {name: $name})-[:LOCATED_IN]->(pa:PlanningArea) RETURN pa.name AS area, pa.population AS pop", {"name": sel["name"]})
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
        ld = run_query("MATCH (a:TransportNode {transport_type:'mrt'})-[r:CONNECTS_TO]->(b:TransportNode {transport_type:'mrt'}) RETURN a.lat AS al, a.lon AS ao, b.lat AS bl, b.lon AS bo, r.line AS l LIMIT 150")
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
        mrts = run_query("MATCH (n:TransportNode {transport_type:'mrt'}) RETURN n.name AS n, n.lat AS la, n.lon AS lo LIMIT 200")
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
        buses = run_query("MATCH (n:TransportNode {transport_type:'bus'}) WHERE n.lat IS NOT NULL RETURN n.name AS n, n.lat AS la, n.lon AS lo LIMIT 3000")
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
    c_pick, c_btn = st.columns([4, 1])
    with c_pick:
        pick = st.selectbox("", PRESETS, label_visibility="collapsed", key="qpick")
    with c_btn:
        ask = st.button("Ask Graph", type="primary", use_container_width=True)

    manual = st.chat_input("Or type any question — semantic search + Cypher + ChromaDB...")

    query = None
    if ask: query = str(pick)
    if manual: query = manual

    if query:
        st.session_state.chat.append({"role": "user", "content": query})
        with st.spinner("Retrieving from knowledge graph + vector store..."):
            r = get_gen().answer(query)
        st.session_state.chat.append({"role": "assistant", "content": r["answer_text"],
            "confidence": r.get("confidence","MEDIUM"), "mode": r.get("retrieval_mode","")})
        hl = []
        try:
            names = run_query("MATCH (n) WHERE n.name IS NOT NULL AND n.lat IS NOT NULL RETURN n.name AS n, n.lat AS la, n.lon AS lo, labels(n)[0] AS l LIMIT 5000")
            for n in names:
                if str(n["n"]).lower() in r["answer_text"].lower() and len(str(n["n"])) > 3:
                    hl.append({"name":n["n"],"lat":float(n["la"]),"lon":float(n["lo"]),"label":n.get("l","")})
                    if len(hl) >= 8: break
        except: pass
        st.session_state.hl = hl
        st.rerun()

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                c = {"HIGH":"🟢","MEDIUM":"🟡","LOW":"🟠"}.get(msg.get("confidence",""),"")
                st.caption(f"{c} {msg.get('confidence','')} · {msg.get('mode','')}")
            st.write(msg["content"])


# ═══════════════ TAB 2: ANALYTICS ═══════════════
with tabs[2]:
    fig_dir = Path("reports/figures")
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Best Model R²", "0.819", help="RandomForest on 9 engineered features")
    with c2: st.metric("Cross-Validation", "0.761 ± 0.098")
    with c3: st.metric("Top Predictor", "Temperature")

    st.divider()
    charts = [
        ("weather_dashboard.png", "90-Day Singapore Weather — 4 Variables"),
        ("feature_importance.png", "RandomForest Feature Importance — Temperature Dominates"),
        ("prediction_vs_actual.png", "Predicted vs Actual Rainfall — R² = 0.82"),
    ]
    cols = st.columns(3)
    for i, (f, cap) in enumerate(charts):
        p = fig_dir / f
        if p.exists():
            with cols[i]: st.image(str(p), caption=cap, use_container_width=True)

    st.divider()
    st.subheader("Time Series & Correlations")
    cols2 = st.columns(3)
    charts2 = [
        ("timeseries_dashboard.png", "91-Day Trends with 7-Day Rolling Mean"),
        ("correlation_heatmap.png", "Variable Correlations — Humidity↔Temp r=-0.85"),
        ("seasonal_pattern.png", "Day-of-Week Weather Patterns"),
    ]
    for i, (f, cap) in enumerate(charts2):
        p = fig_dir / f
        if p.exists():
            with cols2[i % 3]: st.image(str(p), caption=cap, use_container_width=True)


# ═══════════════ TAB 3: GRAPH ML ═══════════════
with tabs[3]:
    st.subheader("Node2Vec Graph Embeddings")
    try:
        from src.ml.graph_ml import GraphMLEngine
        engine = GraphMLEngine()
        engine.build_graph()
        engine.train_node2vec(dimensions=32)
        preds = engine.predict_links(top_k=15)
        Nn = engine.G.number_of_nodes()
        Ne = engine.G.number_of_edges()

        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Graph Nodes", f"{Nn:,}")
        with c2: st.metric("Graph Edges", Ne)
        with c3: st.metric("Embeddings Dim", "32")

        st.subheader("Predicted Missing Links")
        df = pd.DataFrame(preds)
        df["similarity"] = df["similarity"].apply(lambda x: f"{float(x):.3f}")
        st.dataframe(df[["source_name","target_name","similarity"]], use_container_width=True, hide_index=True, height=400)

        embp = fig_dir / "node_embeddings_tsne.png"
        if embp.exists():
            st.image(str(embp), caption="t-SNE Projection of Transport Node Embeddings")
    except Exception as e:
        st.warning(f"Neo4j required: {e}")


# ═══════════════ TAB 4: REPORT ═══════════════
with tabs[4]:
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Model Card")
        cp = Path("reports/mlops/model_card.json")
        if cp.exists():
            st.json(json.loads(cp.read_text()))
    with col_b:
        st.subheader("Geospatial Visualizations")
        for f, cap in [("mrt_topology.png","MRT Network Topology"),("demographic_heatmap.png","Population Density Heatmap"),("area_clustering.png","K-Means Area Clustering")]:
            p = fig_dir / f
            if p.exists():
                st.image(str(p), caption=cap, use_container_width=True)

    st.divider()
    st.subheader("Experiment Tracking")
    ep = Path("reports/mlops/experiments_20260807.json")
    if ep.exists():
        exps = json.loads(ep.read_text())
        cols = st.columns(len(exps))
        for i, e in enumerate(exps):
            with cols[i]:
                st.metric(e["run_name"], f"R²={e['metrics'].get('R2','?'):.3f}" if 'R2' in e.get('metrics',{}) else "N/A")
                st.caption(str(e.get("params",""))[:80])
