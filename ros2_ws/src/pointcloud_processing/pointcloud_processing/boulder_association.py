"""Boulder landmark association with CLIPPER on a traverse CSV.

A traverse CSV has rows ``frame,x,y,z,detections,boulders``: the rover position
and a list of boulder centroids, both in the world frame. The CSV has no
heading, so the rover frame here is world-aligned and centred on the rover;
detections are converted to it by subtracting the logged position. Odometry is
the logged motion with a scale error, noise, and yaw drift added. At each
detection frame the detections are placed with the predicted pose, paired with
nearby map landmarks, filtered by CLIPPER, and a yaw-and-translation correction
is fitted to CLIPPER's inliers. Unmatched detections start new landmarks.

Ground-truth boulder identities come from clustering every detection of the
traverse in the world frame. A pair is correct when the detection and the
landmark carry the same identity.
"""

from __future__ import annotations

import ast
import csv
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from pointcloud_processing.clipper_association import ClipperConfig, associate


@dataclass
class Traverse:
    frames: np.ndarray
    truth: np.ndarray
    boulders: list[np.ndarray]


def load_traverse(path: str | Path) -> Traverse:
    csv.field_size_limit(1 << 30)
    frames, poses, boulders = [], [], []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            frames.append(int(row["frame"]))
            poses.append([float(row["x"]), float(row["y"]), float(row["z"])])
            points = ast.literal_eval(row["boulders"]) if row["boulders"].strip() else []
            boulders.append(np.asarray(points, dtype=np.float64).reshape(-1, 3))
    return Traverse(np.asarray(frames), np.asarray(poses, dtype=np.float64).reshape(-1, 3), boulders)


def cluster_identities(points: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Greedy running-centroid clustering. Returns a label per point and the centroids."""
    sums: list[np.ndarray] = []
    counts: list[int] = []
    labels = np.empty(points.shape[0], dtype=np.int64)
    for index, point in enumerate(points):
        best, best_distance = -1, radius
        for label, (total, count) in enumerate(zip(sums, counts)):
            distance = float(np.linalg.norm(total / count - point))
            if distance < best_distance:
                best, best_distance = label, distance
        if best < 0:
            sums.append(point.copy())
            counts.append(1)
            best = len(sums) - 1
        else:
            sums[best] += point
            counts[best] += 1
        labels[index] = best
    centroids = np.array([total / count for total, count in zip(sums, counts)]).reshape(-1, 3)
    return labels, centroids


def yaw_matrix(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def fit_yaw_translation(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray]:
    """Least-squares yaw and 3D translation mapping ``source`` onto ``target``."""
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    a, b = source[:, :2] - source_mean[:2], target[:, :2] - target_mean[:2]
    yaw = float(np.arctan2(np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]), np.sum(a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1])))
    translation = target_mean - yaw_matrix(yaw) @ source_mean
    return yaw, translation


class LandmarkMap:
    def __init__(self) -> None:
        self.sums: list[np.ndarray] = []
        self.counts: list[int] = []
        self.votes: list[dict[int, int]] = []

    def __len__(self) -> int:
        return len(self.sums)

    def positions(self) -> np.ndarray:
        if not self.sums:
            return np.zeros((0, 3))
        return np.array([total / count for total, count in zip(self.sums, self.counts)])

    def identity(self, index: int) -> int:
        votes = self.votes[index]
        return max(votes, key=votes.get)

    def identities(self) -> np.ndarray:
        return np.array([self.identity(index) for index in range(len(self))], dtype=np.int64)

    def update(self, index: int, point: np.ndarray, identity: int) -> None:
        self.sums[index] += point
        self.counts[index] += 1
        self.votes[index][identity] = self.votes[index].get(identity, 0) + 1

    def add(self, point: np.ndarray, identity: int) -> None:
        self.sums.append(point.copy())
        self.counts.append(1)
        self.votes.append({identity: 1})


@dataclass
class BoulderConfig:
    seed: int = 0
    scale_error: float = 0.03
    translation_noise: float = 0.02
    yaw_drift_deg_per_m: float = 2.0
    yaw_noise_deg_per_m: float = 0.5
    identity_radius: float = 0.15
    gate_radius: float = 1.0
    max_candidates_per_detection: int = 5
    max_rmse: float = 0.1
    max_correction_yaw_deg: float = 5.0
    max_correction_m: float = 0.5
    merge_radius: float = 0.25
    correct: bool = True
    clipper: ClipperConfig = field(default_factory=lambda: ClipperConfig(sigma=0.02, epsilon=0.05, mindist=0.05, min_inliers=3))


@dataclass
class FrameResult:
    """One detection frame. Pair arrays index ``detections_*`` rows and ``landmarks`` rows.

    ``landmarks`` and ``landmark_identities`` are the map before this frame's
    update, which is what the candidates were drawn against.
    """

    row: int
    frame: int
    truth: np.ndarray
    position: np.ndarray
    yaw: float
    odometry_position: np.ndarray
    odometry_yaw: float
    identities: np.ndarray
    detections_predicted: np.ndarray
    detections_corrected: np.ndarray
    landmarks: np.ndarray
    landmark_identities: np.ndarray
    candidates: np.ndarray
    kept: np.ndarray
    correct: np.ndarray
    accepted: bool
    reason: str
    record: dict


class BoulderSession:
    """Steps through a traverse; ``step`` advances to the next detection frame."""

    def __init__(self, traverse: Traverse, config: BoulderConfig) -> None:
        self.traverse = traverse
        self.config = config
        rows = [index for index, points in enumerate(traverse.boulders) if points.shape[0] > 0]
        self.detection_rows = rows
        points = np.concatenate([traverse.boulders[index] for index in rows]) if rows else np.zeros((0, 3))
        labels, self.identity_centroids = cluster_identities(points, config.identity_radius)
        self.detection_count = int(points.shape[0])
        self.row_identities: dict[int, np.ndarray] = {}
        offset = 0
        for index in rows:
            count = traverse.boulders[index].shape[0]
            self.row_identities[index] = labels[offset : offset + count]
            offset += count
        self.reset()

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.config.seed)
        self.row = 0
        self.position = self.traverse.truth[0].copy()
        self.yaw = 0.0
        self.odometry_position = self.traverse.truth[0].copy()
        self.odometry_yaw = 0.0
        self.landmarks = LandmarkMap()
        self.seen_identities: set[int] = set()
        self.records: list[dict] = []
        self.totals = dict.fromkeys(
            ("clipper_pairs", "clipper_correct", "nn_pairs", "nn_correct", "matchable", "matched_correct"), 0
        )

    @property
    def done(self) -> bool:
        return self.row >= len(self.traverse.frames)

    def _advance_odometry(self, index: int) -> None:
        config = self.config
        truth = self.traverse.truth
        step = truth[index] - truth[index - 1]
        distance = float(np.linalg.norm(step[:2]))
        spread = np.sqrt(distance + 1e-12)
        measured = step * (1.0 + config.scale_error) + self.rng.normal(0.0, config.translation_noise * spread, 3)
        turn = np.deg2rad(config.yaw_drift_deg_per_m) * distance + self.rng.normal(
            0.0, np.deg2rad(config.yaw_noise_deg_per_m) * spread
        )
        self.position = self.position + yaw_matrix(self.yaw) @ measured
        self.yaw += turn
        self.odometry_position = self.odometry_position + yaw_matrix(self.odometry_yaw) @ measured
        self.odometry_yaw += turn

    def step(self) -> FrameResult | None:
        """Advance odometry to the next detection frame and process it. None at the end."""
        while not self.done:
            index = self.row
            self.row += 1
            if index > 0:
                self._advance_odometry(index)
            if index in self.row_identities:
                return self._process(index)
        return None

    def run(self) -> list[FrameResult]:
        results = []
        while (result := self.step()) is not None:
            results.append(result)
        return results

    def _process(self, index: int) -> FrameResult:
        config = self.config
        landmarks = self.landmarks
        started = time.perf_counter()
        truth = self.traverse.truth[index]
        identities = self.row_identities[index]
        local = self.traverse.boulders[index] - truth
        predicted = local @ yaw_matrix(self.yaw).T + self.position
        map_points = landmarks.positions()
        map_identities = landmarks.identities()
        candidates = np.zeros((0, 2), dtype=np.int32)
        if len(landmarks):
            tree = cKDTree(map_points)
            pairs = []
            for source, point in enumerate(predicted):
                distances, targets = tree.query(point, k=min(config.max_candidates_per_detection, len(landmarks)))
                for distance, target in zip(np.atleast_1d(distances), np.atleast_1d(targets)):
                    if distance <= config.gate_radius:
                        pairs.append((source, int(target)))
            candidates = np.asarray(pairs, dtype=np.int32).reshape(-1, 2)

        clipper_started = time.perf_counter()
        result = associate(predicted, map_points, candidates, config.clipper)
        clipper_s = time.perf_counter() - clipper_started
        inliers = result.inliers
        inlier_set = {(int(source), int(target)) for source, target in inliers}
        kept = np.array([(int(s), int(t)) in inlier_set for s, t in candidates], dtype=bool)
        correct = (
            map_identities[candidates[:, 1]] == identities[candidates[:, 0]]
            if candidates.shape[0]
            else np.zeros(0, dtype=bool)
        )

        accepted = False
        reason = str(result.diagnostics.get("reason", ""))
        correction_yaw, rmse, shift = 0.0, float("nan"), 0.0
        corrected_position = self.position
        if result.count >= config.clipper.min_inliers and len(np.unique(inliers[:, 0])) >= 3:
            correction_yaw, correction_translation = fit_yaw_translation(
                predicted[inliers[:, 0]], map_points[inliers[:, 1]]
            )
            aligned = predicted[inliers[:, 0]] @ yaw_matrix(correction_yaw).T + correction_translation
            rmse = float(np.sqrt(np.mean(np.sum((aligned - map_points[inliers[:, 1]]) ** 2, axis=1))))
            corrected_position = yaw_matrix(correction_yaw) @ self.position + correction_translation
            shift = float(np.linalg.norm(corrected_position - self.position))
            if rmse > config.max_rmse:
                reason = "rmse_gate"
            elif abs(np.rad2deg(correction_yaw)) > config.max_correction_yaw_deg or shift > config.max_correction_m:
                reason = "correction_gate"
            else:
                accepted, reason = True, "accepted"
        if accepted and config.correct:
            self.position = corrected_position
            self.yaw += correction_yaw
        corrected = local @ yaw_matrix(self.yaw).T + self.position

        matchable = np.array([identity in self.seen_identities for identity in identities], dtype=bool)
        assigned = np.full(local.shape[0], -1, dtype=np.int64)
        clipper_correct = 0
        if accepted:
            for source, target in inliers:
                if assigned[source] < 0:
                    assigned[source] = target
                    clipper_correct += int(map_identities[target] == identities[source])
        clipper_pairs = int(np.sum(assigned >= 0))
        nn_pairs = nn_correct = 0
        if len(landmarks):
            tree = cKDTree(map_points)
            for source in np.flatnonzero(assigned < 0):
                distance, target = tree.query(corrected[source])
                if distance <= config.merge_radius:
                    assigned[source] = int(target)
                    nn_pairs += 1
                    nn_correct += int(map_identities[target] == identities[source])
        matched_correct = int(
            sum(assigned[s] >= 0 and map_identities[assigned[s]] == identities[s] for s in range(local.shape[0]))
        )
        for source in range(local.shape[0]):
            if assigned[source] >= 0:
                landmarks.update(int(assigned[source]), corrected[source], int(identities[source]))
            else:
                landmarks.add(corrected[source], int(identities[source]))
        self.seen_identities.update(int(identity) for identity in identities)

        for key, value in (
            ("clipper_pairs", clipper_pairs),
            ("clipper_correct", clipper_correct),
            ("nn_pairs", nn_pairs),
            ("nn_correct", nn_correct),
            ("matchable", int(matchable.sum())),
            ("matched_correct", matched_correct),
        ):
            self.totals[key] += value
        error = float(np.linalg.norm(self.position - truth))
        odometry_error = float(np.linalg.norm(self.odometry_position - truth))
        record = {
            "frame": int(self.traverse.frames[index]),
            "detections": int(local.shape[0]),
            "matchable": int(matchable.sum()),
            "candidates": int(candidates.shape[0]),
            "correct_candidates": int(correct.sum()),
            "clipper_inliers": int(result.count),
            "clipper_inliers_correct": int(np.sum(kept & correct)),
            "clipper_pairs": clipper_pairs,
            "clipper_correct": clipper_correct,
            "nn_pairs": nn_pairs,
            "nn_correct": nn_correct,
            "accepted": bool(accepted),
            "reason": reason,
            "fit_rmse": rmse,
            "correction_yaw_deg": float(np.rad2deg(correction_yaw)) if accepted else 0.0,
            "correction_m": shift if accepted else 0.0,
            "landmarks": len(landmarks),
            "pose": self.position.round(4).tolist(),
            "yaw_deg": float(np.rad2deg(self.yaw)),
            "truth": truth.round(4).tolist(),
            "translation_error": error,
            "odometry_translation_error": odometry_error,
            "odometry_yaw_deg": float(np.rad2deg(self.odometry_yaw)),
            "clipper_s": clipper_s,
            "total_s": time.perf_counter() - started,
        }
        self.records.append(record)
        return FrameResult(
            row=index,
            frame=int(self.traverse.frames[index]),
            truth=truth.copy(),
            position=self.position.copy(),
            yaw=self.yaw,
            odometry_position=self.odometry_position.copy(),
            odometry_yaw=self.odometry_yaw,
            identities=identities.copy(),
            detections_predicted=predicted,
            detections_corrected=corrected,
            landmarks=map_points,
            landmark_identities=map_identities,
            candidates=candidates,
            kept=kept,
            correct=correct,
            accepted=accepted,
            reason=reason,
            record=record,
        )

    def summary(self) -> dict:
        records = self.records
        totals = self.totals

        def ratio(numerator: float, denominator: float) -> float | None:
            return float(numerator / denominator) if denominator else None

        errors = np.array([row["translation_error"] for row in records])
        odometry_errors = np.array([row["odometry_translation_error"] for row in records])
        candidates = sum(row["candidates"] for row in records)
        correct_candidates = sum(row["correct_candidates"] for row in records)
        inliers = sum(row["clipper_inliers"] for row in records)
        inliers_correct = sum(row["clipper_inliers_correct"] for row in records)
        rejected = [row for row in records if not row["accepted"]]
        return {
            "rows": len(self.traverse.frames),
            "detection_frames": len(records),
            "detections": self.detection_count,
            "ground_truth_boulders": int(self.identity_centroids.shape[0]),
            "final_landmarks": len(self.landmarks),
            "path_length_m": float(np.sum(np.linalg.norm(np.diff(self.traverse.truth, axis=0), axis=1))),
            "accepted": len(records) - len(rejected),
            "rejections": {
                reason: sum(1 for row in rejected if row["reason"] == reason)
                for reason in sorted({row["reason"] for row in rejected})
            },
            "candidate_pairs": candidates,
            "candidate_pairs_correct": correct_candidates,
            "clipper_inlier_pairs": inliers,
            "clipper_inlier_pairs_correct": inliers_correct,
            "clipper_inlier_precision": ratio(inliers_correct, inliers),
            "clipper_inlier_recall_of_correct_candidates": ratio(inliers_correct, correct_candidates),
            "clipper_precision": ratio(totals["clipper_correct"], totals["clipper_pairs"]),
            "clipper_recall": ratio(totals["clipper_correct"], totals["matchable"]),
            "nearest_neighbour_precision": ratio(totals["nn_correct"], totals["nn_pairs"]),
            "overall_recall": ratio(totals["matched_correct"], totals["matchable"]),
            "matchable_detections": totals["matchable"],
            "final_translation_error_m": float(errors[-1]) if errors.size else None,
            "max_translation_error_m": float(errors.max()) if errors.size else None,
            "rmse_translation_error_m": float(np.sqrt(np.mean(errors**2))) if errors.size else None,
            "final_yaw_error_deg": records[-1]["yaw_deg"] if records else None,
            "odometry_final_translation_error_m": float(odometry_errors[-1]) if errors.size else None,
            "odometry_max_translation_error_m": float(odometry_errors.max()) if errors.size else None,
            "odometry_final_yaw_error_deg": records[-1]["odometry_yaw_deg"] if records else None,
            "mean_clipper_s": float(np.mean([row["clipper_s"] for row in records])) if records else None,
            "mean_total_s": float(np.mean([row["total_s"] for row in records])) if records else None,
        }
