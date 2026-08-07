# UrbanGraph-SG

> **GraphRAG-powered urban knowledge navigator for Singapore**

UrbanGraph-SG integrates multi-source Singapore open data (LTA, NEA, SingStat, OneMap, HDB) into a unified knowledge graph, enabling natural-language question answering with source-attributed, evidence-bound responses and interactive graph visualization.

## Architecture

```
Multi-source Open Data → Ingestion Agents (×6) → Processing → GraphRAG Pipeline
                                                                    ↓
                    Streamlit UI ← Answer Generation ← Neo4j Knowledge Graph
                         ↕
                   PyVis Visualization
```

## Quick Start

```bash
# Clone and setup
git clone <repo-url>
cd urbangraph-sg
cp .env.example .env   # edit with your API keys
make setup

# Ingest data (requires API keys)
make ingest-all

# Process and build knowledge graph
make process
make index
make load-graph

# Launch UI
make up
# Streamlit: http://localhost:8502
# Neo4j Browser: http://localhost:7474
```

## Tech Stack

- **Knowledge Graph**: Neo4j 5.x
- **GraphRAG**: Microsoft GraphRAG + Custom Singapore prompts
- **LLM**: DeepSeek / OpenAI / Ollama (local)
- **UI**: Streamlit + PyVis
- **Infra**: Docker Compose, GitHub Actions

## Project Status

🚧 **Stage 0 — Scaffolding complete.** Moving to data ingestion agents.

## License

MIT
