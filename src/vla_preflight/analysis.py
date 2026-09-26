"""Full numeric profiling, duplicate-aware splits and train-only normalization."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import __version__
from .audit import audit
from .dataset import Dataset, rows
from .workflow_io import atomic_json, digest_json, file_hash, fingerprint, new_artifact_dir


class Moments:
    def __init__(self, dim, *, seed=7, capacity=8192):
        self.count = 0
        self.mean = np.zeros(dim)
        self.m2 = np.zeros(dim)
        self.minimum = np.full(dim, np.inf)
        self.maximum = np.full(dim, -np.inf)
        self.sample = np.zeros((capacity, dim))
        self.rng = np.random.default_rng(seed)

    def add(self, value):
        value = np.asarray(value, dtype=np.float64)
        if value.shape != self.mean.shape or not np.isfinite(value).all():
            raise ValueError("Statistics require finite vectors with a consistent shape")
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)
        self.minimum = np.minimum(self.minimum, value)
        self.maximum = np.maximum(self.maximum, value)
        position = (
            self.count - 1 if self.count <= len(self.sample) else self.rng.integers(self.count)
        )
        if position < len(self.sample):
            self.sample[position] = value

    def result(self):
        if not self.count:
            raise ValueError("No valid rows for statistics")
        q = np.quantile(self.sample[: min(self.count, len(self.sample))], [0.01, 0.5, 0.99], axis=0)
        return {
            "count": self.count,
            "mean": self.mean.tolist(),
            "std": np.sqrt(self.m2 / self.count).tolist(),
            "min": self.minimum.tolist(),
            "max": self.maximum.tolist(),
            "q01": q[0].tolist(),
            "median": q[1].tolist(),
            "q99": q[2].tolist(),
            "quantile_method": "exact"
            if self.count <= len(self.sample)
            else f"reservoir estimate, n={len(self.sample)}",
        }


def profile_dataset(root: Path, *, seed=7, preview_points=200):
    ds = Dataset(root)
    dims = {
        key: ds.features[key]["shape"][0]
        for key in ("action", "observation.state")
        if key in ds.features
    }
    moments = {key: Moments(dim, seed=seed) for key, dim in dims.items()}
    # Numeric series are kept per episode for motion/lag diagnostics, not images.
    collected = defaultdict(list)
    for path in ds.data_files(set(ds.episodes)):
        for row in rows(path, ["episode_index", "frame_index", "timestamp", *dims]):
            if row.get("episode_index") in ds.episodes:
                collected[row["episode_index"]].append(row)
    episodes, groups = [], defaultdict(list)
    for ep in sorted(ds.episodes):
        records = collected[ep]
        valid, invalid = [], 0
        for row in records:
            try:
                vectors = {k: np.asarray(row[k], dtype=float) for k in dims}
                if any(
                    x.shape != (dims[k],) or not np.isfinite(x).all() for k, x in vectors.items()
                ):
                    raise ValueError
                for key, values in vectors.items():
                    moments[key].add(values)
                valid.append(row)
            except (KeyError, TypeError, ValueError):
                invalid += 1
        item = {
            "episode": ep,
            "frames": len(records),
            "invalid_rows": invalid,
            "tasks": ds.episodes[ep].get("tasks", []),
            "duration_seconds": len(records) / ds.fps,
        }
        if valid:
            arrays = {k: np.asarray([r[k] for r in valid], dtype=np.float64) for k in dims}
            action = arrays["action"]
            digest = hashlib.sha256()
            digest.update(digest_json(item["tasks"]).encode())
            for key in sorted(arrays):
                digest.update(key.encode())
                digest.update(np.asarray(arrays[key], dtype="<f8").tobytes())
            # Only complete episodes participate in duplicate grouping.
            numeric_hash = digest.hexdigest() if invalid == 0 else f"invalid-{ep}"
            groups[numeric_hash].append(ep)
            differences = np.linalg.norm(np.diff(action, axis=0), axis=1)
            selected = np.unique(
                np.linspace(0, len(valid) - 1, min(preview_points, len(valid))).astype(int)
            )
            item.update(
                {
                    "numeric_hash": numeric_hash,
                    "action_mean": action.mean(axis=0).tolist(),
                    "action_std": action.std(axis=0).tolist(),
                    "action_change_mean": float(differences.mean()) if len(differences) else 0,
                    "action_change_max": float(differences.max()) if len(differences) else 0,
                    "unchanged_action_fraction": float(np.mean(differences < 1e-8))
                    if len(differences)
                    else 1,
                    "preview_frames": [valid[i]["frame_index"] for i in selected],
                    "preview": {k: x[selected].tolist() for k, x in arrays.items()},
                }
            )
            if "observation.state" in arrays and arrays["observation.state"].shape == action.shape:
                state = arrays["observation.state"]
                candidate_scores = []
                for lag in range(-3, 4):
                    start, stop = max(0, -lag), min(len(action), len(action) - lag)
                    a, s = action[start:stop], state[start + lag : stop + lag]
                    if len(a) < 5:
                        continue
                    a, s = a - a.mean(axis=0), s - s.mean(axis=0)
                    den = np.sqrt(np.sum(a * a) * np.sum(s * s))
                    if den > 1e-12:
                        candidate_scores.append(
                            {"lag_frames": lag, "correlation": float(np.sum(a * s) / den)}
                        )
                item["lag_probe"] = candidate_scores
                item["lag_probe_note"] = (
                    "Exploratory action[t]/state[t+lag] correlation; not a synchronization verdict"
                )
        else:
            groups[f"invalid-{ep}"].append(ep)
        episodes.append(item)
    return {
        "schema": "vla-preflight.profile/1",
        "tool_version": __version__,
        "dataset": ds.root.name,
        "version": ds.version,
        "fps": ds.fps,
        "features": ds.features,
        "total_frames": sum(len(x) for x in collected.values()),
        "statistics": {k: m.result() for k, m in moments.items() if m.count},
        "episodes": episodes,
        "duplicate_groups": [x for x in groups.values() if len(x) > 1],
        "limits": [
            "Duplicates compare numeric state/action and episode task labels, not images.",
            "Lag correlations do not infer physical latency or establish a repair.",
            "Preview curves are decimated; summary metrics use all valid numeric rows.",
        ],
    }


def split_episodes(profile, *, validation_fraction=0.2, seed=7):
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    groups = defaultdict(list)
    for ep in profile["episodes"]:
        groups[ep.get("numeric_hash", f"episode-{ep['episode']}")].append(ep["episode"])
    items = list(groups.values())
    if len(items) < 2:
        raise ValueError("Need at least two distinct numeric episode groups for a held-out split")
    np.random.default_rng(seed).shuffle(items)
    count = max(1, min(len(items) - 1, round(len(items) * validation_fraction)))
    val = sorted(ep for group in items[:count] for ep in group)
    train = sorted(ep for group in items[count:] for ep in group)
    return {
        "seed": seed,
        "requested_validation_fraction": validation_fraction,
        "actual_validation_fraction": len(val) / (len(train) + len(val)),
        "train": train,
        "validation": val,
        "method": (
            "seeded numeric-duplicate-group split; no stratification or semantic deduplication"
        ),
    }


def train_statistics(ds, episodes):
    selected = set(episodes)
    if not selected or not selected <= ds.episodes.keys():
        raise ValueError("Statistics require known, nonempty episode IDs")
    fields = {
        k: Moments(ds.features[k]["shape"][0])
        for k in ("action", "observation.state")
        if k in ds.features
    }
    for path in ds.data_files(selected):
        for row in rows(path, ["episode_index", *fields]):
            if row["episode_index"] in selected:
                for key, moment in fields.items():
                    moment.add(row[key])
    return {key: value.result() for key, value in fields.items()}


def prepare(root: Path, destination: Path, *, validation_fraction=0.2, seed=7, video="exists"):
    report = audit(root, video=video)
    if not report.complete or report.status == "failed":
        raise ValueError("Dataset has preflight errors; inspect an audit report before preparation")
    profile = profile_dataset(root, seed=seed)
    split = split_episodes(profile, validation_fraction=validation_fraction, seed=seed)
    ds = Dataset(root)
    normal = {
        "schema": "vla-preflight.normalization/1",
        "mode": "MEAN_STD",
        "epsilon": 1e-6,
        "fit_episodes": split["train"],
        "features": train_statistics(ds, split["train"]),
    }
    source = fingerprint(ds, videos=video != "skip")
    destination = new_artifact_dir(destination, root)
    atomic_json(destination / "profile.json", profile)
    atomic_json(destination / "split.json", split)
    atomic_json(destination / "normalization.json", normal)
    report.write(destination / "preflight.json", dataset_root=root)
    manifest = {
        "schema": "vla-preflight.prepare/1",
        "tool_version": __version__,
        "complete": True,
        "dataset_root": str(ds.root),
        "source": source,
        "artifact_hashes": {
            name: file_hash(destination / name)
            for name in ("profile.json", "split.json", "normalization.json")
        },
    }
    atomic_json(destination / "manifest.json", manifest)
    return manifest
