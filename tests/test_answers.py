"""Test answer generation with fixed queries."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from src.generation.answer_generator import AnswerGenerator

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
    ans = result["answer_text"][:250]
    print(f"[{mode}/{conf}] Q: {q}")
    print(f"  {ans}")
    print()
