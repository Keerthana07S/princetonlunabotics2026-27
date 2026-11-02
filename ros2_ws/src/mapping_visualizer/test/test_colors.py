"""Color mapping tests."""

from __future__ import annotations

import numpy as np

from mapping_visualizer.colors import DYNAMIC, FRESH, HIGH, LOW, STALE, age_colors, confidence_colors


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


def test_age_colors_fade_from_fresh_to_stale() -> None:
    colors = age_colors(np.array([0.0, 5.0, 30.0, 1.0]), np.array([False, False, False, True]), horizon=10.0)
    assert np.allclose(colors[0], FRESH)
    assert np.allclose(colors[1], 0.5 * (FRESH + STALE))
    assert np.allclose(colors[2], STALE)
    assert np.allclose(colors[3], DYNAMIC)
