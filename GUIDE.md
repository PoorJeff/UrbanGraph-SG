# UrbanGraph-SG — Complete Project Guide

> Last updated: 2026-08-07  
> Status: **All 10 enhancement features complete. 5-tab Streamlit dashboard live.**

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Current Status — What's Done](#2-current-status--whats-done)
3. [Architecture](#3-architecture)
4. [How to Use](#4-how-to-use)
5. [CS/AI Domain Coverage](#5-csai-domain-coverage)
6. [Remaining Work — Priority Order](#6-remaining-work--priority-order)
7. [File Map](#7-file-map)
8. [Quality Checklist](#8-quality-checklist)

---

## 1. Project Overview

**UrbanGraph-SG** is a GraphRAG-powered urban knowledge navigator for Singapore. It ingests multi-source open data (LTA, NEA, SingStat, OneMap, HDB), builds a Neo4j knowledge graph (5,536 nodes, 10,964 edges), and enables natural-language Q&A with source attribution, confidence labeling, and interactive map visualization.

**Target audience**: Graduate school admissions (NUS, NTU, SMU) for CS/AI programs.  
**GitHub**: https://github.com/PoorJeff/UrbanGraph-SG

---

## 2. Current Status — What's Done

### 2.1 Core Pipeline (6 Stages)

| # | Stage | What It Does | Key Files |
|---|---|---|---|
| 0 | Project Scaffold | pyproject.toml, Makefile, Docker, CI, YAML configs | `pyproject.toml`, `makefile`, `docker-compose.yml` |
| 1 | Data Ingestion (6 Agents) | LTA/ NEA / SingStat / OneMap / HDB / Calendar → Parquet | `src/ingestion/lta.py`, `nea.py`, `onemap.py`, etc. |
| 2 | Data Processing (5 Modules) | Time normalization, spatial validation, entity resolution, GraphRAG formatting, quality report | `src/processing/` |
| 3 | GraphRAG Indexing | Entity/relationship extraction via DeepSeek, community detection, LLM summarization | `src/graphrag/` |
| 4 | Neo4j + Retrieval | Graph storage, local/global/Cypher search agents | `src/graph/`, `src/retrieval/` |
| 5 | Answer + UI | Answer generation + Streamlit 5-tab dashboard + Folium map | `src/generation/`, `src/ui/` |
| 6 | Evaluation + Docs | README, pytest, answer quality audit, data audit | `README.md`, `tests/` |

### 2.2 ML/AI Enhancement Modules (8 Modules)

| # | Module | Domain | Technology |
|---|---|---|---|
| 1 | NLP Semantic Search | NLP | `sentence-transformers`, `all-MiniLM-L6-v2` |
| 2 | ChromaDB Vector Store | Vector DB | `chromadb`, persistent embeddings, hybrid search |
| 3 | Weather Prediction | ML | `RandomForest R²=0.817, CV-R²=0.573` |
| 4 | Time Series Analysis | Stats | Seasonal decomposition, r=-0.85 Humidity↔Temp |
| 5 | CV Visualization | CV / Viz | MRT topology, population heatmap, K-Means clustering |
| 6 | Graph ML Node2Vec | Graph ML | SVD embeddings, link prediction, t-SNE |
| 7 | Multi-Agent System | Agent Architecture | Planner→Retriever→Reasoner→Visualizer |
| 8 | MLOps Pipeline | MLOps | Experiment tracker, model card |

### 2.3 Top 10 Enhancement Features

| # | Feature | What It Does |
|---|---|---|
| P0-1 | **FastAPI REST Backend** | 5 API endpoints: health, stats, query, cypher, entities, presets |
| P0-2 | **Docker One-Click** | `docker-compose up` → Neo4j + Streamlit |
| P0-3 | **4-Layer NER Parser** | Entity Linker → Intent Classifier → Slot Filler → Validator+Executor |
| P1-4 | **Multi-Turn Memory** | "What about Tampines?" reuses previous intent |
| P1-5 | **Path Explanation** | Bishan→Braddell→Toa Payoh→Novena→Newton→Orchard (not just "5 hops") |
| P1-6 | **Temporal KG** | BEFORE chain on WeatherEvents, monthly rainfall comparison |
| P2-7 | **Prophet Forecast** | 7-day rainfall prediction with 80% CI |
| P2-8 | **Cloud Ready** | `requirements.txt`, graceful Neo4j fallback |
| P2-9 | **Optuna Hyperopt** | Automatic hyperparameter search for RandomForest |
| P3-10 | **CI/CD Pipeline** | GitHub Actions: pytest on push to main |

### 2.4 Streamlit Dashboard (5 Tabs)

| Tab | Content |
|---|---|
| 🗺️ **Explore** | Full-width map, search bar, 4 toggle layers, radius query, info panel, legend |
| 💬 **Query** | Categorized preset buttons, chat with confidence %, expandable sources, mini charts |
| 📊 **Analytics** | Live model metrics, 3-model comparison cards, 6 charts, correlation insights |
| 🔬 **Graph ML** | Node2Vec training, link predictions with explanations, graph stats, t-SNE |
| 📋 **Report** | System health (Neo4j/ChromaDB/LLM/Memory), model card, experiment timeline, export |

### 2.5 Data Inventory

| Data Source | Records | Notes |
|---|---|---|
| LTA Bus Stops | 5,207 | Full Singapore |
| LTA Bus Routes | 26,863 | All active routes |
| OneMap Planning Areas | 55 | GeoJSON polygons |
| MRT Stations | 137 | Static fallback (7 lines) |
| NEA Weather | 91 days, 10M+ readings | Daily aggregates |
| HDB Resale | ~100K transactions | 26 towns, 2 years |
| Population | 55 areas | SingStat estimates |
| Holidays | 26 | 2025-2026 |

### 2.6 Neo4j Knowledge Graph

| Node Label | Count | Relationship | Count |
|---|---|---|---|
| TransportNode | 5,344 | LOCATED_IN | 5,395 |
| PlanningArea | 55 | CONNECTS_TO | 130 |
| EntityCommunity | 93 | CONTAINS | 5,439 |
| Holiday | 26 | BEFORE | 3 |
| WeatherStation | 14 | | |
| WeatherEvent | 4 | | |
| **Total** | **5,536** | **Total** | **10,967** |

---

## 3. Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    DATA INGESTION                        │
│  LTA(🚇Bus)  NEA(🌧️)  SingStat(👥)  OneMap(🗺️)  HDB(🏠)  │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│                  DATA PROCESSING                         │
│  Time → Spatial → Entity Resolution → GraphRAG Format    │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│                GRAPHRAG INDEXING                         │
│  Entity Extract → Relationship Extract → Community → LLM │
│  (DeepSeek)      (DeepSeek)          Detection   Summary │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│              NEO4J KNOWLEDGE GRAPH                       │
│  5,536 nodes · 10,967 edges · 7 MRT lines · 55 areas    │
└──────────┬──────────────────────────┬───────────────────┘
           ▼                          ▼
┌─────────────────────┐   ┌──────────────────────────────┐
│    RETRIEVAL        │   │       ML / AI MODULES         │
│ Local / Global /    │   │ RandomForest · Prophet        │
│ Cypher / Semantic   │   │ Node2Vec · K-Means           │
│ (ChromaDB)          │   │ Time Series · MLOps           │
└──────────┬──────────┘   └──────────┬───────────────────┘
           ▼                          ▼
┌─────────────────────────────────────────────────────────┐
│              ANSWER GENERATION                           │
│  4-Layer NER Parser → Cypher/Semantic/Local → LLM       │
│  Multi-turn memory · Path explanation · Source citation  │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│                STREAMLIT DASHBOARD                       │
│  Explore │ Query │ Analytics │ Graph ML │ Report        │
│  (Folium map + 4 layers + radius search)                │
└─────────────────────────────────────────────────────────┘
         │                    │
         ▼                    ▼
┌─────────────┐    ┌──────────────────┐
│  FastAPI     │    │  Docker Compose   │
│  REST API    │    │  One-Click Deploy │
└─────────────┘    └──────────────────┘
```

---

## 4. How to Use

### 4.1 Quick Start (Local)

```bash
# 1. Start Neo4j
cd C:\Users\Jzh20\Desktop\neo4j-community-5.26.4\bin
neo4j.bat console

# 2. Start Streamlit (new terminal)
cd C:\Users\Jzh20\Desktop\UrbanGraph-SG
streamlit run src/ui/streamlit_app.py
# → Open http://localhost:8502

# 3. Start FastAPI (optional, new terminal)
cd C:\Users\Jzh20\Desktop\UrbanGraph-SG
uvicorn src.api.server:app --port 8080
# → Open http://localhost:8080/docs for Swagger UI
```

### 4.2 One-Click Launch

```bash
# Windows
scripts\launch.bat

# Or manually
start "Neo4j" /MIN "C:\Users\Jzh20\Desktop\neo4j-community-5.26.4\bin\neo4j.bat" console
streamlit run src\ui\streamlit_app.py --server.port=8502
```

### 4.3 API Endpoints

| Method | URL | Description |
|---|---|---|
| GET | `/api/health` | Health check |
| GET | `/api/stats` | Knowledge graph statistics |
| POST | `/api/query` | Natural language Q&A `{"question": "..."}` |
| POST | `/api/cypher` | Execute Cypher `{"query": "MATCH ... RETURN ..."}` |
| GET | `/api/entities?q=...&limit=5` | Semantic entity search |
| GET | `/api/presets` | List 40 available Cypher presets |

### 4.4 CLI / Python

```python
from src.generation.answer_generator import AnswerGenerator
gen = AnswerGenerator()
result = gen.answer("How many MRT stations are there in total?")
print(result["answer_text"])  # "There are 137 MRT stations in total."
```

### 4.5 Verified Questions (22/22 working)

**Transport — Counts:**
- How many MRT stations are there in total? → 137
- How many bus stops are there in Singapore? → 5,207
- How many stations are on the Circle Line? → 28
- How many MRT stations are in the CBD area? → 32

**Transport — Lines:**
- Which MRT lines pass through Bishan? → CCL, NSL
- Which MRT lines pass through Jurong East? → EWL, NSL
- Which MRT lines serve Woodlands? → NSL

**Transport — Lists:**
- List all MRT stations in Orchard → [list]
- Which MRT stations are in Downtown Core? → [list]

**Transport — Connectivity:**
- Which station has the most connections? → [name]
- Is Bishan station connected to Orchard? → Bishan→Braddell→Toa Payoh→Novena→Newton→Orchard
- How many stations from Jurong East to City Hall? → N hops

**Transport — Bus:**
- List bus stops along Orchard Road → [list]
- How many bus stops are near Orchard MRT? → N

**Population:**
- What is the population of Bedok? → 276,000
- What is the population of Tampines? → 270,000
- What is the population of Punggol? → 195,000
- Which planning area has the largest population? → Bedok
- Which areas have the smallest population? → [list]

**Housing:**
- Which area has the highest HDB resale prices? → Bukit Timah
- How many HDB transactions are in the database? → N

**Spatial:**
- Which planning area is Bedok MRT in? → [area]

---

## 5. CS/AI Domain Coverage

| Domain | Demonstrated Through |
|---|---|
| **NLP / LLM** | Semantic search, NER pipeline, answer generation, multi-turn memory, prompt engineering |
| **Vector DB** | ChromaDB persistent embeddings, hybrid graph+vector retrieval |
| **Knowledge Graphs** | Neo4j 5.x, Cypher queries (40+ presets), community detection, temporal chains |
| **Machine Learning** | RandomForest/GradientBoosting/LinearRegression, feature engineering, Chronological split, CV-based selection |
| **Graph ML** | Node2Vec embeddings (SVD), link prediction, t-SNE visualization |
| **Time Series** | Prophet forecast, seasonal decomposition, rolling statistics, correlation analysis |
| **Computer Vision / Viz** | MRT topology graph, population heatmap, K-Means, prediction vs actual plots |
| **Multi-Agent** | Planner→Retriever→Reasoner→Visualizer pipeline |
| **Data Engineering** | 6 ingestion agents, ETL pipeline, Parquet, data validation |
| **Full-Stack** | Streamlit dashboard, FastAPI REST backend, Folium interactive maps |
| **MLOps** | Experiment tracking, model cards, Optuna hyperparameter optimization |
| **DevOps** | Docker Compose, GitHub Actions CI/CD, launch scripts |

---

## 6. Remaining Work — Priority Order

### Phase 1: Deploy (☐ not started)

| # | Task | Effort | How |
|---|---|---|---|
| 1 | Streamlit Cloud deployment | 15 min | Push to GitHub → https://share.streamlit.io |
| 2 | Neo4j AuraDB free tier | 10 min | https://neo4j.com/cloud/aura/ → free instance |
| 3 | Update .env with remote URIs | 5 min | Replace localhost with cloud URIs |
| 4 | Verify deployment works | 10 min | Test all 5 tabs on cloud URL |

### Phase 2: Application Materials (☐ not started)

| # | Task | Effort | Details |
|---|---|---|---|
| 5 | Chinese resume entry | 15 min | 3-4 line bullet point, SKILLS section |
| 6 | English resume entry | 15 min | Same, translated |
| 7 | Personal Statement paragraph | 20 min | "From UrbanFlow-AU to UrbanGraph-SG to FYP" narrative |
| 8 | README screenshots | 15 min | Screenshot each tab → save to `screenshots/` |
| 9 | Demo video (optional) | 1 hr | 2-minute Loom/OBS recording walking through all tabs |

### Phase 3: Polish (☐ not started)

| # | Task | Effort | Details |
|---|---|---|---|
| 10 | NEA full 2-year data | 30 min | Run `make ingest-nea` with full date range |
| 11 | HDB data load | 15 min | Ensure resale_prices in Neo4j |
| 12 | Fix Launch.bat | 5 min | Ensure it starts Neo4j + Streamlit reliably |
| 13 | Final pytest run | 5 min | Verify 37/37 pass |

### Phase 4: Optional Enhancements (☐ not started)

| # | Task | Effort | Impact |
|---|---|---|---|
| 14 | Grafana dashboard integration | 1 hr | Low |
| 15 | Slack/Telegram bot | 2 hr | Medium |
| 16 | PDF report export | 1 hr | Low |
| 17 | Multilingual support (Chinese) | 2 hr | Medium |

---

## 7. File Map

```
urbangraph-sg/
├── GUIDE.md                    ← THIS FILE
├── README.md                   ← Project README (368 lines)
├── ROADMAP.md                  ← Enhancement roadmap (8 modules + 10 features)
├── agent.md                    ← Agent specification (original architecture)
├── UrbanGraph-SG-report.md     ← Project charter & acceptance criteria
├── pyproject.toml              ← Python project config
├── Makefile                    ← Build automation (make ingest-all, make process, etc.)
├── .env.example                ← Environment template
├── requirements.txt            ← pip install -r requirements.txt
├── docker-compose.yml          ← Docker Compose (Neo4j + Streamlit)
├── configs/                    ← 7 YAML config files
│   ├── data_sources.yaml       ← API endpoints & auth
│   ├── entity_types.yaml       ← Knowledge graph entity schema
│   ├── relationship_types.yaml ← Knowledge graph relationship schema
│   ├── graphrag_config.yaml    ← GraphRAG pipeline parameters
│   ├── search_config.yaml      ← Search & retrieval parameters
│   ├── preset_questions.yaml   ← 23 verified preset questions
│   └── schema_mapping.yaml     ← Raw data → standard field mapping
├── src/
│   ├── config.py               ← Config loader (env + YAML)
│   ├── ingestion/              ← Stage 1: 6 data agents
│   │   ├── base.py             ← Retry logic, pagination, manifest
│   │   ├── lta.py              ← LTA bus stops/services/routes
│   │   ├── nea.py              ← NEA weather (7 endpoints)
│   │   ├── singstat.py         ← Population statistics
│   │   ├── onemap.py           ← Planning areas, MRT stations
│   │   ├── hdb.py              ← HDB resale prices
│   │   ├── calendar.py         ← Holidays, school terms
│   │   └── ingest_all.py       ← Run all 6 agents
│   ├── processing/             ← Stage 2: 5 processing modules
│   │   ├── time_normalizer.py  ← SGT timezone
│   │   ├── spatial_validator.py← Singapore bounds + spatial join
│   │   ├── entity_resolution.py← Entity alignment + global IDs
│   │   ├── graphrag_formatter.py← CSV format for GraphRAG
│   │   ├── quality_reporter.py ← Data quality report
│   │   └── run_all.py          ← Pipeline orchestrator
│   ├── graphrag/               ← Stage 3: GraphRAG indexing
│   │   ├── llm_client.py       ← DeepSeek client (retry, tracking)
│   │   ├── entity_extractor.py ← LLM entity extraction
│   │   ├── relationship_extractor.py ← LLM relationship inference
│   │   ├── community_detector.py← Louvain community detection
│   │   ├── summarizer.py       ← LLM community summarization
│   │   └── prompts/            ← 3 prompt YAML templates
│   ├── graph/                  ← Stage 4: Neo4j storage
│   │   ├── neo4j_client.py     ← Connection, query, batch
│   │   ├── schema.py           ← Constraints, indexes
│   │   └── write_to_neo4j.py   ← Batch data loader
│   ├── retrieval/              ← Stage 4: Retrieval agents
│   │   ├── local_search.py     ← Entity-centric subgraph search
│   │   ├── global_search.py    ← Community map-reduce search
│   │   ├── cypher_agent.py     ← NL→Cypher translation (40+ presets)
│   │   ├── semantic_search.py  ← Embedding-based search
│   │   ├── vector_store.py     ← ChromaDB persistent index
│   │   └── query_parser.py     ← ★ 4-layer NER pipeline
│   ├── generation/             ← Stage 5: Answer generation
│   │   ├── answer_generator.py ← LLM answer + source + confidence
│   │   └── prompts/answer_prompt.yaml
│   ├── ui/                     ← Stage 5: Streamlit dashboard
│   │   └── streamlit_app.py    ← ★ 5-tab dashboard
│   ├── api/                    ← P0-1: REST API
│   │   └── server.py           ← FastAPI server (6 endpoints)
│   ├── agent/                  ← P0-7: Multi-agent
│   │   └── orchestrator.py     ← Planner→Retriever→Reasoner→Visualizer
│   ├── ml/                     ← ML modules
│   │   ├── weather_predictor.py← RandomForest prediction
│   │   ├── visualization.py    ← Matplotlib charts
│   │   ├── timeseries_analysis.py ← Seasonal decomposition
│   │   ├── geo_visualization.py← MRT topology, heatmap, clustering
│   │   ├── graph_ml.py         ← Node2Vec embeddings
│   │   ├── hyperopt.py         ← Optuna hyperparameter search
│   │   ├── prophet_forecast.py ← Prophet 7-day prediction
│   │   └── mlops.py            ← Experiment tracker + model card
│   └── evaluation/             ← Evaluation framework
├── tests/                      ← Test files
│   ├── unit/                   ← 4 test modules (37 tests)
│   ├── test_answers.py         ← Answer quality test
│   ├── test_ner.py             ← NER pipeline test
│   └── comprehensive_test.py   ← Full system test
├── data/                       ← Data directory (parquet excluded from git)
│   ├── raw/                    ← Raw ingested data
│   ├── processed/              ← Processed entities + community texts
│   ├── graphrag/               ← GraphRAG inputs/outputs
│   └── manifests/              ← Ingestion metadata
├── reports/                    ← Generated reports
│   ├── data_quality/           ← Quality reports
│   ├── figures/                ← 12+ ML/CV/forecast charts
│   ├── mlops/                  ← Model cards, experiment logs
│   └── graph/                  ← Schema validation
├── scripts/                    ← Utility scripts
│   ├── demo.py                 ← Interactive CLI demo
│   └── launch.bat              ← One-click Windows launcher
├── docker/                     ← Dockerfiles
├── requirements/               ← Dependency files (4)
└── .github/workflows/          ← CI pipelines
    ├── ci.yml                  ← pytest on push
    └── index-check.yml         ← GraphRAG smoke test
```

---

## 8. Quality Checklist

### ✅ Verified

- [x] pytest: 37/37 tests passing
- [x] Answer quality: 22/22 preset questions correct, 0% fabrication
- [x] ML pipeline: No data leakage (chronological split), best model by CV
- [x] NEA data: All weather metrics within normal tropical ranges
- [x] Neo4j integrity: 5,536 nodes, 10,967 edges, no duplicate issues
- [x] Cypher presets: 36/40 working (4 parameter-missing in audit only)
- [x] Streamlit: All 5 tabs rendering correctly
- [x] FastAPI: All 6 endpoints returning 200
- [x] ChromaDB: 5,439 entities indexed
- [x] K-Means: 4 meaningful clusters (Silhouette score computed)
- [x] Multi-turn memory: "What about Tampines?" correctly reuses intent
- [x] Path explanation: Shows full station chain, not just hop count
- [x] Git: No API keys in repo, .env excluded, 20+ clean commits

### ☐ Remaining

- [ ] Streamlit Cloud / Railway deployment
- [ ] Neo4j AuraDB remote connection
- [ ] README screenshots (5 tabs)
- [ ] Resume entries (Chinese + English)
- [ ] Personal Statement narrative paragraph
- [ ] Demo video (optional)
- [ ] Full 2-year NEA data ingestion
- [ ] Grafana dashboard (optional)
