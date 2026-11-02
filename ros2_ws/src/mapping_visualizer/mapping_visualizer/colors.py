"""Color rules for the live map. Independent of Viser."""

from __future__ import annotations

import numpy as np

LOW = np.array([0.15, 0.35, 0.95])
HIGH = np.array([0.85, 0.85, 0.85])
DYNAMIC = np.array([1.0, 0.45, 0.05])
SCAN = np.array([0.1, 0.85, 0.25])


def confidence_colors(
    confidence: np.ndarray,
    dynamic: np.ndarray,
    low_confidence: float,
    high_confidence: float,
) -> np.ndarray:
    """Blend low and high colors, then paint dynamic points separately."""
    values = np.asarray(confidence, dtype=np.float64).reshape(-1)
    moving = np.asarray(dynamic, dtype=bool).reshape(-1)
    if values.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64)
    span = max(high_confidence - low_confidence, 1e-6)
    weight = np.clip((values - low_confidence) / span, 0.0, 1.0)
    colors = (1.0 - weight)[:, None] * LOW + weight[:, None] * HIGH
    colors[moving] = DYNAMIC
    return colors
