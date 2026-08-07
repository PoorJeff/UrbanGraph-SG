"""Unit tests for Cypher agent."""
from src.retrieval.cypher_agent import execute, run_preset, list_presets, FORBIDDEN


class TestCypherSecurity:
    def test_delete_blocked(self):
        r = execute("MATCH (n) DELETE n")
        assert "error" in r
        assert "Forbidden" in r["error"]

    def test_drop_blocked(self):
        r = execute("DROP CONSTRAINT foo")
        assert "error" in r

    def test_create_blocked(self):
        r = execute("CREATE (n:Test)")
        assert "error" in r

    def test_set_blocked(self):
        r = execute("MATCH (n) SET n.foo = 1")
        assert "error" in r

    def test_merge_blocked(self):
        r = execute("MERGE (n:Test)")
        assert "error" in r

    def test_read_allowed(self):
        r = execute("MATCH (n) RETURN count(n) AS c")
        assert "error" not in r
        assert "results" in r

    def test_limit_added(self):
        r = execute("MATCH (n) RETURN n")
        assert "LIMIT" in r.get("_query", "") or r.get("count", 0) <= 1000


class TestPresets:
    def test_presets_exist(self):
        presets = list_presets()
        assert len(presets) >= 20
        assert "station_count" in presets
        assert "mrt_lines_bishan" in presets
        assert "largest_population" in presets
        assert "hdb_highest_prices" in presets

    def test_unknown_preset(self):
        r = run_preset("nonexistent_query")
        assert "error" in r


class TestExecute:
    def test_valid_query(self):
        r = execute("RETURN 1 AS test")
        assert r["results"] == [{"test": 1}]
        assert r["count"] == 1
