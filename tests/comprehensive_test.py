"""Comprehensive system test for UrbanGraph-SG.
Tests all question types and identifies what works vs what fails."""

import sys, io, json, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from src.generation.answer_generator import AnswerGenerator
from src.retrieval.cypher_agent import run_preset

gen = AnswerGenerator()

# ── Comprehensive question battery ──
questions = {
    "Transport — MRT stations": [
        "How many MRT stations are there in total?",
        "Which MRT lines pass through Bishan?",
        "Which MRT lines pass through Jurong East?",
        "How many MRT stations are in the CBD area?",
        "List all MRT stations in Orchard",
        "Which MRT stations are in Downtown Core?",
        "What MRT lines serve Woodlands?",
        "How many stations are on the Circle Line?",
    ],
    "Transport — Bus": [
        "How many bus stops are there in Singapore?",
        "List bus stops along Orchard Road",
        "What bus services stop at Jurong East?",
    ],
    "Transport — Connectivity": [
        "Is Bishan station connected to Orchard?",
        "How many stations from Jurong East to City Hall?",
        "Which station has the most connections?",
    ],
    "Population & Demographics": [
        "What is the population of Bedok?",
        "What is the population of Tampines?",
        "What is the population of Punggol?",
        "What is the population of Ang Mo Kio?",
        "Which planning area has the largest population?",
        "Which areas have the smallest population?",
    ],
    "Housing": [
        "What is the average HDB resale price in Punggol?",
        "Which area has the highest HDB prices?",
        "How many HDB transactions are there?",
    ],
    "Weather": [
        "What is the average temperature in Singapore?",
        "How much rain did Singapore get yesterday?",
        "Which month has the most rainfall?",
    ],
    "Spatial": [
        "Which planning area is Bedok MRT in?",
        "How many bus stops are near Orchard MRT?",
        "What planning area is Changi Airport in?",
    ],
    "Edge cases — should say I don't know": [
        "What is the best restaurant in Singapore?",
        "Who is the prime minister of Singapore?",
        "What is the GDP of Singapore?",
        "How do I get from my house to NUS?",
    ],
}

total = 0
passed = 0
results = {}

SEP = "─" * 70

for category, qs in questions.items():
    print(f"\n{'='*70}")
    print(f"  {category}")
    print(f"{'='*70}")
    for q in qs:
        total += 1
        result = gen.answer(q)
        ans = result["answer_text"].strip()
        mode = result.get("retrieval_mode", "?")
        conf = result.get("confidence", "?")

        # Heuristic: does the answer seem useful?
        is_dont_know = "don't have enough data" in ans.lower()
        has_numbers = bool(re.search(r'\d[\d,]*\d', ans))
        is_empty = len(ans) < 20
        quality = "✅" if (has_numbers and not is_dont_know) or (not is_dont_know and len(ans) > 50) else "⚠️" if not is_dont_know and len(ans) > 30 else "❌"

        if quality == "✅":
            passed += 1

        results[q] = {"quality": quality, "mode": mode, "conf": conf, "ans": ans[:200]}

        print(f"  {quality} [{mode}] {q}")
        print(f"     → {ans[:180].replace(chr(10),' ')}")
        print()

# ── Summary ──
print(f"\n{'='*70}")
print(f"  RESULTS: {passed}/{total} passed ({passed*100//total}%)")
print(f"{'='*70}")

print(f"\n✅ Works perfectly:")
for q, r in results.items():
    if r["quality"] == "✅":
        print(f"   {q}")

print(f"\n⚠️ Partially works / needs tuning:")
for q, r in results.items():
    if r["quality"] == "⚠️":
        print(f"   {q} → {r['ans'][:100]}...")

print(f"\n❌ Broken / No data:")
for q, r in results.items():
    if r["quality"] == "❌":
        print(f"   {q} → {r['ans'][:100]}...")

# Save results
with open("tests/test_results.json", "w", encoding="utf-8") as f:
    json.dump({"passed": passed, "total": total, "results": results}, f, indent=2, ensure_ascii=False)
print(f"\nResults saved to tests/test_results.json")

# LLM cost
stats = gen.llm.get_stats()
print(f"\nLLM Cost: ${stats['total_cost_estimate']:.4f} ({stats['total_calls']} calls, {stats['total_tokens']:,} tokens)")
