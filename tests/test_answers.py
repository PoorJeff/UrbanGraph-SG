"""Quick test of answer generation quality."""
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
]

for q in questions:
    result = gen.answer(q)
    print(f"Q: {q}")
    print(f"  Mode: {result['retrieval_mode']}, Confidence: {result.get('confidence','?')}")
    print(f"  A: {result['answer_text'][:300]}")
    print()
