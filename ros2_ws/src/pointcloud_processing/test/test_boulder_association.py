"""Boulder landmarks: CLIPPER pairs should be correct and should undo odometry drift."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("clipperpy")

from pointcloud_processing.boulder_association import (  # noqa: E402
    BoulderConfig,
    BoulderSession,
    fit_yaw_translation,
    load_traverse,
    yaw_matrix,
)


def _write_traverse(path, seed: int = 3) -> None:
    rng = np.random.default_rng(seed)
    field = np.column_stack([rng.uniform(-2.0, 2.0, 60), rng.uniform(-1.0, 12.0, 60), rng.uniform(-0.05, 0.05, 60)])
    lines = ["frame,x,y,z,detections,boulders"]
    for frame in range(1, 1001):
        rover = np.array([0.0, 0.01 * frame, 0.15])
        seen = []
        if frame % 10 == 0:
            offset = field - rover
            visible = (offset[:, 1] > 1.0) & (offset[:, 1] < 4.0) & (np.abs(offset[:, 0]) < 2.0)
            seen = field[visible] + rng.normal(0.0, 0.01, (int(visible.sum()), 3))
        boulders = "[" + ", ".join(f"({p[0]}, {p[1]}, {p[2]})" for p in seen) + "]"
        lines.append(f'{frame},{rover[0]},{rover[1]},{rover[2]},{len(seen)},"{boulders}"')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_yaw_fit_recovers_a_known_motion() -> None:
    rng = np.random.default_rng(0)
    source = rng.uniform(-3.0, 3.0, (6, 3))
    target = source @ yaw_matrix(0.2).T + np.array([0.5, -0.3, 0.1])
    yaw, translation = fit_yaw_translation(source, target)
    assert yaw == pytest.approx(0.2, abs=1e-9)
    np.testing.assert_allclose(translation, [0.5, -0.3, 0.1], atol=1e-9)


def test_clipper_pairs_are_correct_and_reduce_drift(tmp_path) -> None:
    csv_path = tmp_path / "traverse.csv"
    _write_traverse(csv_path)
    traverse = load_traverse(csv_path)
    session = BoulderSession(traverse, BoulderConfig(yaw_drift_deg_per_m=3.0))
    results = session.run()
    summary = session.summary()

    assert len(results) == 100
    assert all(result.candidates.shape[0] == result.kept.shape[0] == result.correct.shape[0] for result in results)
    assert summary["accepted"] > 50
    assert summary["clipper_inlier_precision"] > 0.95
    assert summary["final_translation_error_m"] < 0.5 * summary["odometry_final_translation_error_m"]
