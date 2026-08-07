"""Unit tests for answer generator."""
from src.generation.answer_generator import AnswerGenerator


class TestDirectPreset:
    """Test that _try_direct_preset works for known questions."""

    def setup_method(self):
        self.gen = AnswerGenerator()

    def test_total_mrt(self):
        r = self.gen._try_direct_preset("How many MRT stations are there in total?")
        assert r is not None
        assert r["retrieval_mode"] == "cypher"
        assert "137" in r["answer_text"]

    def test_circle_line(self):
        r = self.gen._try_direct_preset("How many stations are on the Circle Line?")
        assert r is not None
        assert "28" in r["answer_text"]

    def test_bus_stops(self):
        r = self.gen._try_direct_preset("How many bus stops are there in Singapore?")
        assert r is not None
        assert "5,207" in r["answer_text"]

    def test_bishan_lines(self):
        r = self.gen._try_direct_preset("Which MRT lines pass through Bishan?")
        assert r is not None
        assert "CCL" in r["answer_text"] or "Circle Line" in r["answer_text"]

    def test_cbd_mrt(self):
        r = self.gen._try_direct_preset("How many MRT stations are in the CBD area?")
        assert r is not None
        assert "32" in r["answer_text"]

    def test_largest_population(self):
        # "largest population" should match the direct preset
        r = self.gen.answer("Which planning area has the largest population?")
        assert r is not None
        assert "Bedok" in r["answer_text"] or "276" in r["answer_text"]

    def test_highest_hdb(self):
        r = self.gen._try_direct_preset("Which area has the highest HDB prices?")
        assert r is not None
        assert "Bukit Timah" in r["answer_text"] or "717" in r["answer_text"]

    def test_unknown_question_returns_none(self):
        r = self.gen._try_direct_preset("What is the best restaurant in Singapore?")
        assert r is None

    def test_bishan_orchard_path(self):
        r = self.gen._try_direct_preset("Is Bishan station connected to Orchard?")
        assert r is not None
        assert "5" in r["answer_text"] or len(r["answer_text"]) > 50
