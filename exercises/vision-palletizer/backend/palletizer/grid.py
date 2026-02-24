"""
Palletizing Grid Calculations
============================

Calculate place positions for boxes in an N×M grid pattern.
Grid fills row-by-row starting from the origin.
"""

from typing import List, Tuple


def calculate_place_positions(
    rows: int,
    cols: int,
    box_size_mm: Tuple[float, float, float],
    pallet_origin_mm: Tuple[float, float, float],
    spacing_mm: float = 10.0,
) -> List[Tuple[float, float, float]]:
    """
    Calculate TCP positions for placing boxes in a grid pattern.

    Positions are the centre of each box cell. The grid fills row-by-row
    (all columns in row 0, then all columns in row 1, etc.).

    X direction = columns (box width + spacing)
    Y direction = rows    (box depth + spacing)
    Z is constant (pallet surface = origin Z)

    Args:
        rows: Number of rows (N)
        cols: Number of columns (M)
        box_size_mm: (width, depth, height) of each box in mm
        pallet_origin_mm: (x, y, z) position of the first box placement
        spacing_mm: Gap between adjacent boxes (default 10mm)

    Returns:
        List of (x, y, z) TCP target positions, ordered for row-by-row filling.
    """
    width, depth, _height = box_size_mm
    ox, oy, oz = pallet_origin_mm

    positions: List[Tuple[float, float, float]] = []

    for row in range(rows):
        for col in range(cols):
            x = ox + col * (width + spacing_mm)
            y = oy + row * (depth + spacing_mm)
            z = oz
            positions.append((x, y, z))

    return positions
