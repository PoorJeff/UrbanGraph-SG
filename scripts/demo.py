#!/usr/bin/env python
"""UrbanGraph-SG Interactive Demo.

Demonstrates all three retrieval modes and the knowledge graph.
Run: python scripts/demo.py
"""

import sys, io
sys.path.insert(0, ".")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from src.generation.answer_generator import AnswerGenerator
from src.retrieval.cypher_agent import run_preset, list_presets
from src.graph.neo4j_client import run_query

SEP = "=" * 70

def main():
    print(f"\n{SEP}")
    print("  UrbanGraph-SG  Interactive Demo")
    print(f"{SEP}\n")

    # === KNOWLEDGE GRAPH STATS ===
    print("[1] Knowledge Graph Summary")
    print("-" * 40)
    stats = run_query("MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt ORDER BY cnt DESC")
    for s in stats:
        print(f"  {s['label']:<20} {s['cnt']:>6}")
    rels = run_query("MATCH ()-[r]->() RETURN count(r) AS total")
    print(f"  {'Relationships':<20} {rels[0]['total']:>6}")
    print(f"  Total nodes: {sum(s['cnt'] for s in stats):,}")
    print()

    # === PRESET CYPHER QUERIES ===
    print("[2] Cypher Query Presets")
    print("-" * 40)
    presets = list_presets()
    for pid in presets:
        r = run_preset(pid)
        if r.get("results"):
            print(f"  {pid}:")
            for row in r["results"][:3]:
                print(f"    {row}")
        else:
            print(f"  {pid}: {r.get('error','no results')}")
    print()

    # === ANSWER GENERATION ===
    print("[3] Natural Language Q&A (GraphRAG)")
    print("-" * 40)

    gen = AnswerGenerator()
    questions = [
        "What is the population of Bedok?",
        "How many MRT stations are there in total?",
        "Which MRT lines pass through Bishan?",
        "Which areas have the most MRT stations?",
        "How many MRT stations are in the CBD area?",
        "What is the population of Tampines?",
    ]

    for q in questions:
        result = gen.answer(q)
        mode = result.get("retrieval_mode", "?")
        conf = result.get("confidence", "?")
        emoji = {"HIGH": "GREEN", "MEDIUM": "YELLOW", "LOW": "ORANGE"}.get(conf, "?")
        print(f"  Q: {q}")
        print(f"  [{mode}/{emoji}] {result['answer_text'].strip()}")
        print()

    # === LLM STATS ===
    stats = gen.llm.get_stats()
    print("[4] LLM Cost Report")
    print("-" * 40)
    print(f"  Model:    {gen.llm.model}")
    print(f"  Calls:    {stats['total_calls']}")
    print(f"  Tokens:   {stats['total_tokens']:,}")
    print(f"  Cost:     ${stats['total_cost_estimate']:.4f}")
    print()

    print(f"{SEP}")
    print("  Demo complete! Try Streamlit UI: streamlit run src/ui/streamlit_app.py")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
