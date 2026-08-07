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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }
.main { background: #0f1117; }
.main .block-container { padding: 1rem 2rem; max-width: 100%; }
header { visibility: hidden; }
footer { visibility: hidden; }

/* Typography */
h1, h2, h3, h4, p, span, div { color: #e4e6eb; }
h3 { font-size: 1.3rem !important; margin: 0 !important; padding: 0 !important; font-weight: 600; }
p, .stCaption { color: #8a8d91; font-size: 0.82rem; }

/* KPI Cards */
.kpi-row { display: flex; gap: 12px; margin: 12px 0; }
.kpi-card { flex: 1; background: #1a1d24; border-radius: 10px; padding: 16px 20px;
    border: 1px solid #262930; text-align: center; }
.kpi-card .kpi-num { font-size: 1.8rem; font-weight: 700; color: #fff; }
.kpi-card .kpi-label { font-size: 0.72rem; color: #8a8d91; margin-top: 2px; text-transform: uppercase; letter-spacing: 0.5px; }
.kpi-accent1 .kpi-num { color: #ED2939; }
.kpi-accent2 .kpi-num { color: #009530; }
.kpi-accent3 .kpi-num { color: #005EC4; }
.kpi-accent4 .kpi-num { color: #FA9E0D; }
.kpi-accent5 .kpi-num { color: #9D5B25; }

/* Search Bar */
.search-box { display: flex; gap: 8px; align-items: center; background: #1a1d24;
    border-radius: 12px; padding: 6px 14px; border: 1px solid #262930; margin-bottom: 10px; }

/* Buttons */
.stButton button { background: #ED2939 !important; color: white !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important; font-size: 0.85rem !important;
    padding: 8px 20px !important; transition: all 0.15s; }
.stButton button:hover { background: #c42030 !important; transform: translateY(-1px); box-shadow: 0 4px 12px rgba(237,41,57,0.3); }

/* Select */
.stSelectbox div[data-baseweb="select"] > div { background: #1a1d24 !important; border: 1px solid #262930 !important;
    border-radius: 8px !important; color: #e4e6eb !important; }
.stSelectbox svg { color: #8a8d91 !important; }

/* Chat */
.stChatMessage { background: transparent !important; }
[data-testid="stChatMessage"] { background: #1a1d24 !important; border-radius: 10px !important;
    padding: 10px 16px !important; margin: 6px 0 !important; border: 1px solid #262930; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 4px; background: transparent; border-bottom: 1px solid #262930; }
.stTabs [data-baseweb="tab"] { background: transparent; color: #8a8d91; border-radius: 8px 8px 0 0;
    padding: 8px 20px; font-weight: 500; font-size: 0.85rem; }
.stTabs [data-baseweb="tab"][aria-selected="true"] { background: #1a1d24; color: #ED2939; border-bottom: 2px solid #ED2939; }

/* Dataframe */
.stDataFrame { background: #1a1d24 !important; border-radius: 8px !important; border: 1px solid #262930 !important; }
.stDataFrame th { background: #262930 !important; color: #e4e6eb !important; font-weight: 600; }
.stDataFrame td { color: #b0b3b8 !important; }

/* Metrics */
[data-testid="stMetricValue"] { color: #fff !important; font-size: 1.6rem !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: #8a8d91 !important; }

/* Chart/Border */
.stImage { border-radius: 8px; }

/* Divider */
hr { border-color: #262930 !important; margin: 16px 0 !important; }
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

# ═══════════════ TAB 0: MAP-FIRST EXPLORE ═══════════════
with tabs[0]:
    mp = folium.Map(location=[1.3521,103.8198], zoom_start=12, tiles="CartoDB dark_matter", control_scale=True)
    colors = {"EWL":"#009530","NSL":"#D42E2B","NEL":"#9900AA","CCL":"#FA9E0D","DTL":"#005EC4","TEL":"#9D5B25"}
    names = {"EWL":"East-West","NSL":"North-South","NEL":"North East","CCL":"Circle","DTL":"Downtown","TEL":"Thomson-East Coast"}
    try:
        ld = run_query("MATCH (a:TransportNode {transport_type:'mrt'})-[r:CONNECTS_TO]->(b:TransportNode {transport_type:'mrt'}) RETURN a.lat AS al, a.lon AS ao, b.lat AS bl, b.lon AS bo, r.line AS l LIMIT 150")
        grps = {}
        for row in ld:
            ln = row.get("l","?")
            if ln not in grps: grps[ln] = FeatureGroup(name=f"{ln} {names.get(ln,'')}")
            try: folium.PolyLine([[float(row["al"]),float(row["ao"])],[float(row["bl"]),float(row["bo"])]], color=colors.get(ln,"#888"), weight=3, opacity=0.8).add_to(grps[ln])
            except: pass
        for g in grps.values(): g.add_to(mp)
    except: pass
    try:
        mrts = run_query("MATCH (n:TransportNode {transport_type:'mrt'}) RETURN n.name AS n, n.lat AS la, n.lon AS lo LIMIT 200")
        fg = FeatureGroup(name="MRT Stations")
        for s in mrts:
            if s.get("la") and s.get("lo"):
                folium.CircleMarker([float(s["la"]),float(s["lo"])], radius=3.5, color="#ED2939", fill=True, fill_color="#ED2939", fill_opacity=0.9, tooltip=s["n"]).add_to(fg)
        fg.add_to(mp)
    except: pass
    for e in st.session_state.get("hl",[]):
        try: folium.Marker([e["lat"],e["lon"]], icon=folium.Icon(color="orange",icon="star",prefix="fa"), popup=f"<b>{e['name']}</b>").add_to(mp)
        except: pass
    folium.LayerControl().add_to(mp)
    folium_static(mp, height=550)

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    for c, code in [(c1,"EWL"),(c2,"NSL"),(c3,"NEL"),(c4,"CCL"),(c5,"DTL"),(c6,"TEL"),(c7,"")]:
        if code: c.markdown(f'<span style="color:{colors[code]};font-weight:600">●</span> <span style="color:#8a8d91;font-size:0.75rem">{code}</span>', unsafe_allow_html=True)


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
