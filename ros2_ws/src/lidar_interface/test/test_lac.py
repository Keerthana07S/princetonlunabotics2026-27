"""LAC log reader against the example recording in the repository."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("lunarloc")

from lidar_interface.lac import LacLog  # noqa: E402

LOG = Path(__file__).resolve().parents[4] / "software" / "pointclouds" / "example (2).lac"


@pytest.mark.skipif(not LOG.exists(), reason="example LAC log is not present")
def test_example_log_has_front_stereo_pairs_and_a_mount() -> None:
    log = LacLog(str(LOG))
    pairs = [frame for frame in log.frames if frame.has_stereo]
    assert len(pairs) == 60
    assert np.allclose(log.left_mount[:3, 3], [0.2798, 0.0811, 0.1307], atol=1e-3)
    assert np.allclose(log.left_mount[:3, :3], np.eye(3))
    left, right = log.stereo_pair(pairs[0].frame)
    assert left.shape == right.shape == (720, 1280)
    assert left.dtype == np.uint8
    assert np.allclose(pairs[0].rover[:3, :3] @ pairs[0].rover[:3, :3].T, np.eye(3))
