"""Color mapping tests."""

from __future__ import annotations

import numpy as np

from mapping_visualizer.colors import DYNAMIC, HIGH, LOW, confidence_colors


def test_dynamic_points_use_the_dynamic_color() -> None:
    colors = confidence_colors(
        np.array([0.1, 2.0, 2.0]),
        np.array([False, False, True]),
        low_confidence=0.2,
        high_confidence=1.5,
    )
    assert np.allclose(colors[0], LOW)
    assert np.allclose(colors[1], HIGH)
    assert np.allclose(colors[2], DYNAMIC)
