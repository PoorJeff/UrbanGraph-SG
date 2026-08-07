"""Test the 4-layer query parser against real questions."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from src.retrieval.query_parser import parse_and_execute, link_entities, classify_intent
from src.generation.answer_generator import AnswerGenerator

gen = AnswerGenerator()

tests = [
    # Transport counts
    "How many MRT stations are there in total?",
    "How many MRT stations are in the CBD area?",
    "How many stations are on the Circle Line?",
    "How many bus stops are there in Singapore?",
    # Population
    "What is the population of Bedok?",
    "What is the population of Tampines?",
    "What is the population of Punggol?",
    "Which planning area has the largest population?",
    "Which areas have the smallest population?",
    # Transport lines
    "Which MRT lines pass through Bishan?",
    "Which MRT lines pass through Jurong East?",
    "Which MRT lines serve Woodlands?",
    "List all MRT stations in Orchard",
    "Which MRT stations are in Downtown Core?",
    # Connectivity
    "Which station has the most connections?",
    "Is Bishan station connected to Orchard?",
    "How many stations from Jurong East to City Hall?",
    # Housing
    "Which area has the highest HDB resale prices?",
    "How many HDB transactions are in the database?",
    # Spatial
    "Which planning area is Bedok MRT in?",
    "List bus stops along Orchard Road",
    # Free-text (best test of NER)
    "Compare the population of Bedok and Tampines",
    "Tell me the population of Ang Mo Kio",
    "What lines serve Bishan MRT station?",
]

passed = 0
failed = 0
for q in tests:
    # Try new parser first
    parsed = parse_and_execute(q)
    r = gen.answer(q)
    ans = r['answer_text']
    has_data = len(ans) > 80 and "don't have enough" not in ans.lower()
    status = "OK" if has_data else "NO_DATA"
    if has_data: passed += 1
    else: failed += 1

    # Show entity linking
    linked = link_entities(q)
    entities_found = [e['name'] for e in linked['entities']]
    intent = classify_intent(q, linked['entities'])

    print(f"[{status}] {q}")
    if not has_data:
        print(f"  NO DATA: {ans[:120]}...")
    if entities_found:
        print(f"  Entities: {entities_found}")
    if intent.get('intent') != 'UNKNOWN':
        print(f"  Intent: {intent['intent']} ({intent['source']})")
    if parsed:
        print(f"  Template: {parsed.get('template','?')} → {parsed['count']} rows")
    print()

print(f"\nPassed: {passed}/{len(tests)} ({passed*100//len(tests)}%)")
