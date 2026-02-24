"""Tests for grid position calculations. Run with: pytest tests/test_grid.py -v"""

import pytest
from palletizer.grid import calculate_place_positions


class TestGridBasic:
    """Basic grid generation tests."""

    def test_single_cell(self):
        positions = calculate_place_positions(
            rows=1, cols=1,
            box_size_mm=(100.0, 100.0, 50.0),
            pallet_origin_mm=(400.0, -200.0, 100.0),
        )
        assert len(positions) == 1
        assert positions[0] == (400.0, -200.0, 100.0)

    def test_2x2_grid_count(self):
        positions = calculate_place_positions(
            rows=2, cols=2,
            box_size_mm=(100.0, 100.0, 50.0),
            pallet_origin_mm=(400.0, -200.0, 100.0),
        )
        assert len(positions) == 4

    def test_3x4_grid_count(self):
        positions = calculate_place_positions(
            rows=3, cols=4,
            box_size_mm=(100.0, 100.0, 50.0),
            pallet_origin_mm=(0.0, 0.0, 0.0),
        )
        assert len(positions) == 12


class TestGridPositions:
    """Verify correct position calculations."""

    def test_2x2_positions_default_spacing(self):
        """2x2 grid with 100mm boxes and 10mm spacing."""
        positions = calculate_place_positions(
            rows=2, cols=2,
            box_size_mm=(100.0, 100.0, 50.0),
            pallet_origin_mm=(0.0, 0.0, 0.0),
            spacing_mm=10.0,
        )
        # Row 0: (0,0,0), (110,0,0)
        # Row 1: (0,110,0), (110,110,0)
        assert positions[0] == (0.0, 0.0, 0.0)
        assert positions[1] == (110.0, 0.0, 0.0)
        assert positions[2] == (0.0, 110.0, 0.0)
        assert positions[3] == (110.0, 110.0, 0.0)

    def test_origin_offset(self):
        """Positions should be offset by pallet origin."""
        positions = calculate_place_positions(
            rows=1, cols=2,
            box_size_mm=(100.0, 100.0, 50.0),
            pallet_origin_mm=(400.0, -200.0, 100.0),
            spacing_mm=10.0,
        )
        assert positions[0] == (400.0, -200.0, 100.0)
        assert positions[1] == (510.0, -200.0, 100.0)

    def test_z_is_constant(self):
        """All Z values should be the pallet origin Z."""
        positions = calculate_place_positions(
            rows=3, cols=3,
            box_size_mm=(50.0, 50.0, 25.0),
            pallet_origin_mm=(100.0, 100.0, 42.0),
        )
        for pos in positions:
            assert pos[2] == 42.0

    def test_zero_spacing(self):
        positions = calculate_place_positions(
            rows=1, cols=3,
            box_size_mm=(100.0, 80.0, 50.0),
            pallet_origin_mm=(0.0, 0.0, 0.0),
            spacing_mm=0.0,
        )
        assert positions[0] == (0.0, 0.0, 0.0)
        assert positions[1] == (100.0, 0.0, 0.0)
        assert positions[2] == (200.0, 0.0, 0.0)


class TestGridOrder:
    """Grid should fill row-by-row."""

    def test_row_by_row_order(self):
        """First row should come before second row."""
        positions = calculate_place_positions(
            rows=2, cols=3,
            box_size_mm=(100.0, 100.0, 50.0),
            pallet_origin_mm=(0.0, 0.0, 0.0),
            spacing_mm=0.0,
        )
        # Row 0: indices 0,1,2  (y=0)
        # Row 1: indices 3,4,5  (y=100)
        for i in range(3):
            assert positions[i][1] == 0.0   # row 0
        for i in range(3, 6):
            assert positions[i][1] == 100.0  # row 1

    def test_columns_increase_x(self):
        positions = calculate_place_positions(
            rows=1, cols=4,
            box_size_mm=(50.0, 50.0, 25.0),
            pallet_origin_mm=(0.0, 0.0, 0.0),
            spacing_mm=5.0,
        )
        for i in range(len(positions) - 1):
            assert positions[i + 1][0] > positions[i][0]
