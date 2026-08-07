"""Unit tests for spatial validator."""
from src.processing.spatial_validator import validate_coordinates, SG_BOUNDS


class TestValidateCoordinates:
    def test_valid_singapore(self):
        valid, reason = validate_coordinates(1.3521, 103.8198)
        assert valid
        assert reason == ""

    def test_invalid_lat_too_low(self):
        valid, reason = validate_coordinates(1.0, 103.8)
        assert not valid
        assert "latitude" in reason

    def test_invalid_lat_too_high(self):
        valid, reason = validate_coordinates(1.6, 103.8)
        assert not valid

    def test_invalid_lon_too_low(self):
        valid, reason = validate_coordinates(1.35, 103.0)
        assert not valid

    def test_invalid_lon_too_high(self):
        valid, reason = validate_coordinates(1.35, 105.0)
        assert not valid

    def test_singapore_bounds(self):
        # Verify bounds cover key locations
        locations = [
            (1.3724, 103.9494),   # Pasir Ris (East)
            (1.3331, 103.7427),   # Jurong East (West)
            (1.4489, 103.8197),   # Sembawang (North)
            (1.2656, 103.8211),   # HarbourFront (South)
            (1.3644, 103.9915),   # Changi (extreme East)
        ]
        for lat, lon in locations:
            valid, _ = validate_coordinates(lat, lon)
            assert valid, f"({lat}, {lon}) should be valid"

    def test_johor_bahru_out_of_bounds(self):
        # Larkin Terminal at 1.4939, just barely outside the 1.49 bound
        valid, _ = validate_coordinates(1.4939, 103.7440)
        assert not valid
