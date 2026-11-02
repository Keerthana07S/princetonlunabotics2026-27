"""Import official CLIPPER and reject planted outlier correspondences."""

from __future__ import annotations

import numpy as np
import clipperpy


def main() -> None:
    source = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.2],
            [0.4, 0.2, 0.8],
        ],
        dtype=np.float64,
    )
    theta = 0.3
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    target = (rotation @ source.T).T + np.array([0.5, -0.2, 0.1])
    rng = np.random.default_rng(0)
    source_all = np.vstack([source, rng.normal(size=(4, 3))])
    target_all = np.vstack([target, rng.normal(size=(4, 3))])
    true_pairs = np.stack([np.arange(6), np.arange(6)], axis=1)
    outlier_pairs = np.array(
        [[6, 7], [7, 6], [8, 9], [9, 8], [6, 9], [7, 8]],
        dtype=np.int32,
    )
    associations = np.vstack([true_pairs, outlier_pairs]).astype(np.int32)

    invariant_params = clipperpy.invariants.EuclideanDistanceParams()
    invariant_params.sigma = 0.01
    invariant_params.epsilon = 0.05
    invariant_params.mindist = 0.0
    invariant = clipperpy.invariants.EuclideanDistance(invariant_params)
    params = clipperpy.Params()
    params.rounding = clipperpy.Rounding.NONZERO
    solver = clipperpy.CLIPPER(invariant, params)
    solver.score_pairwise_consistency(
        np.asfortranarray(source_all.T),
        np.asfortranarray(target_all.T),
        np.asfortranarray(associations),
    )
    solver.solve()
    selected = np.asarray(solver.get_selected_associations())
    print("clipperpy", clipperpy.__version__)
    print("selected", selected.tolist())
    if selected.ndim != 2 or selected.shape[0] < 4:
        raise SystemExit(f"CLIPPER returned too few inliers: {selected}")
    true_hits = np.sum((selected[:, 0] == selected[:, 1]) & (selected[:, 0] < 6))
    if true_hits < 4:
        raise SystemExit(f"CLIPPER did not keep the rigid inliers: {selected}")
    if selected.shape[0] >= associations.shape[0]:
        raise SystemExit("CLIPPER kept every candidate, including planted outliers")
    print("CLIPPER smoke test passed")


if __name__ == "__main__":
    main()
