"""Unit tests for entity resolution."""
import pytest
from src.processing.entity_resolution import normalize_name, levenshtein_distance


class TestNormalizeName:
    def test_lowercase(self):
        assert normalize_name("Orchard MRT Station") == "orchard"

    def test_strip_suffix(self):
        assert normalize_name("Jurong East MRT Station") == "jurong east"
        assert normalize_name("Bishan MRT") == "bishan"
        assert normalize_name("Tampines LRT Station") == "tampines"

    def test_strip_punctuation(self):
        assert normalize_name("St. Joseph's Ch") == "st josephs ch"

    def test_empty(self):
        assert normalize_name("") == ""
        assert normalize_name(None) == ""

    def test_bus_stop_name(self):
        assert normalize_name("Bedok Int") == "bedok int"


class TestLevenshteinDistance:
    def test_same_string(self):
        assert levenshtein_distance("hello", "hello") == 0

    def test_one_edit(self):
        assert levenshtein_distance("orchard", "orchad") == 1

    def test_two_edits(self):
        assert levenshtein_distance("kitten", "sitting") == 3

    def test_empty(self):
        assert levenshtein_distance("", "abc") == 3
        assert levenshtein_distance("abc", "") == 3

    def test_case_sensitive(self):
        assert levenshtein_distance("Orchard", "orchard") == 1

    def test_station_names(self):
        assert levenshtein_distance("dhoby ghaut", "dhoby ghaut stn") == 4
