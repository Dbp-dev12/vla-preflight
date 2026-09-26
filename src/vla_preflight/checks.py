"""Explicit compatibility checks; never infer physical semantics from magnitudes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .contract import Contract, load_json
from .dataset import Dataset, inside
from .report import Report


def check_contract(ds: Dataset, contract: Contract, report: Report):
    for key, expected in (
        ("action", contract.action_dim),
        ("observation.state", contract.state_dim),
    ):
        if expected is not None and ds.features.get(key, {}).get("shape") != [expected]:
            report.add(
                "CONFIG_DIMENSION",
                "error",
                "Dataset and contract dimensions differ.",
                "Use the dataset feature shape or fix the source schema.",
                feature=key,
                expected=[expected],
                actual=ds.features.get(key, {}).get("shape"),
            )
    for key in contract.camera_keys:
        if ds.features.get(key, {}).get("dtype") not in ("image", "video"):
            report.add(
                "CONFIG_CAMERA",
                "error",
                "Required camera is absent or not visual.",
                "Supply the correct dataset camera keys after accounting for remapping.",
                feature=key,
            )
    if contract.fps is not None and not np.isclose(contract.fps, ds.fps):
        report.add(
            "CONFIG_FPS",
            "error",
            "Declared training and dataset frequencies differ.",
            "Verify explicit resampling; do not just relabel the dataset fps.",
            expected=contract.fps,
            actual=ds.fps,
        )
    train, val = contract.train_episodes, contract.validation_episodes
    if train is not None and val is not None and (overlap := sorted(set(train) & set(val))):
        report.add(
            "SPLIT_OVERLAP",
            "error",
            "Training and validation share episode IDs.",
            "Split by disjoint episodes, not individual frames.",
            episodes=overlap[:20],
        )
    for name, ids in (("train", train), ("validation", val)):
        if ids is not None and (missing := sorted(set(ids) - ds.episodes.keys())):
            report.add(
                "SPLIT_UNKNOWN",
                "error",
                "Split refers to nonexistent episodes.",
                "Correct split IDs against meta/episodes.",
                split=name,
                episodes=missing[:20],
            )
    if contract.training and contract.deployment:
        report.coverage["semantic_comparison"] = "explicit declarations only; adapters not executed"
        for key, value in contract.training.model_dump().items():
            other = getattr(contract.deployment, key)
            if value is not None and other is not None and value != other:
                report.add(
                    "SEMANTICS_MISMATCH",
                    "error",
                    "Declared train/deploy semantics differ.",
                    "Verify the adapter and keep units, joint order and action mode consistent.",
                    field=key,
                    training=value,
                    deployment=other,
                )
            elif (value is None) != (other is None):
                report.add(
                    "SEMANTICS_UNKNOWN",
                    "warning",
                    "Semantic field is missing on one side.",
                    "Declare both sides; no physical semantics are guessed.",
                    field=key,
                )
    else:
        report.coverage["semantic_comparison"] = "not checked: two explicit declarations required"
    if contract.training and contract.training.action_names:
        names = ds.features["action"].get("names")
        if isinstance(names, dict) and list(names) == ["motors"]:
            names = names["motors"]
        if isinstance(names, list) and names != contract.training.action_names:
            report.add(
                "ACTION_ORDER",
                "error",
                "Dataset and training action names/order differ.",
                "Confirm the intended joint mapping before starting training.",
                dataset=names,
                training=contract.training.action_names,
            )
        elif not isinstance(names, list):
            report.add(
                "ACTION_ORDER_UNKNOWN",
                "warning",
                "Dataset action names cannot be resolved.",
                "Verify joint order manually; accepted names are a list or a motors list.",
            )


def check_stats(ds: Dataset, contract: Contract | None, report: Report):
    modes = contract.normalization if contract else {}
    path = inside(ds.root, "meta/stats.json")
    if not path.exists():
        required = any(mode != "IDENTITY" for mode in modes.values())
        report.add(
            "STATS_MISSING",
            "error" if required else "warning",
            "Aggregate meta/stats.json is absent.",
            "Provide the statistics actually used by your trainer. v2.1 may aggregate "
            "episodes_stats.jsonl at runtime; this tool does not reproduce that aggregation.",
        )
        report.coverage["statistics"] = "not available"
        return
    try:
        stats = load_json(path)
        if not isinstance(stats, dict):
            raise ValueError("stats must be an object")
    except (OSError, ValueError) as exc:
        report.add(
            "STATS_INVALID",
            "error",
            "Cannot read statistics.",
            "Regenerate statistics.",
            detail=str(exc),
        )
        return
    report.coverage["statistics"] = "stored numeric-vector schema; not recomputed from data"
    required_keys = {
        "MEAN_STD": ("mean", "std"),
        "MIN_MAX": ("min", "max"),
        "QUANTILES": ("q01", "q99"),
        "IDENTITY": (),
    }
    for key in modes:
        if key not in ds.features:
            report.add(
                "NORMALIZATION_FEATURE",
                "error",
                "Normalization refers to an absent feature.",
                "Correct the feature name in the audit contract.",
                feature=key,
            )
    for key, spec in ds.features.items():
        if key not in ("action", "observation.state"):
            continue
        item = stats.get(key, {})
        if not isinstance(item, dict):
            report.add(
                "STATS_INVALID",
                "error",
                "Feature statistics must be an object.",
                "Regenerate statistics.",
                feature=key,
            )
            continue
        for stat in required_keys.get(modes.get(key, "IDENTITY"), ()):
            if stat not in item:
                report.add(
                    "STATS_REQUIRED",
                    "error",
                    "Normalization statistics are missing.",
                    "Compute the statistics required by the configured normalization mode.",
                    feature=key,
                    statistic=stat,
                    mode=modes[key],
                )
        arrays = {}
        for stat in ("mean", "std", "min", "max", "q01", "q99"):
            if stat not in item:
                continue
            try:
                arr = np.asarray(item[stat], dtype=float)
                if list(arr.shape) != spec["shape"] or not np.isfinite(arr).all():
                    raise ValueError("wrong shape or nonfinite values")
                arrays[stat] = arr
            except (TypeError, ValueError):
                report.add(
                    "STATS_INVALID",
                    "error",
                    "Invalid numeric-vector statistics.",
                    "Recompute statistics with the same feature ordering as the dataset.",
                    feature=key,
                    statistic=stat,
                )
        if "std" in arrays and np.any(arrays["std"] < 0):
            report.add(
                "STATS_NEGATIVE_STD",
                "error",
                "Standard deviation is negative.",
                "Regenerate statistics.",
                feature=key,
            )
        if "std" in arrays and np.any(arrays["std"] == 0):
            report.add(
                "STATS_ZERO_STD",
                "warning",
                "Some dimensions have zero standard deviation.",
                "Check fixed joints and the trainer's epsilon handling.",
                feature=key,
            )
        for lower, upper in (("min", "max"), ("q01", "q99")):
            if lower in arrays and upper in arrays and np.any(arrays[lower] > arrays[upper]):
                report.add(
                    "STATS_RANGE",
                    "error",
                    "Statistic lower bound exceeds upper bound.",
                    "Regenerate statistics.",
                    feature=key,
                    lower=lower,
                    upper=upper,
                )


def check_windows(path: Path, ds: Dataset, report: Report):
    """Validate an exported sampler trace, not a hypothetical reconstructed sampler.

    Each row: episode_index, anchor_frame, target_episode_indices,
    target_frame_indices, padding_mask. True masks a repeated endpoint.
    Targets are contiguous and start at anchor_frame (v1 trace convention).
    """
    count = 0
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                episode, anchor = item["episode_index"], item["anchor_frame"]
                if type(episode) is not int or type(anchor) is not int:
                    raise ValueError("episode_index and anchor_frame must be integers")
                length = ds.episodes[episode]["length"]
                eps, frames, masks = (
                    item[k]
                    for k in ("target_episode_indices", "target_frame_indices", "padding_mask")
                )
                if not all(isinstance(x, list) for x in (eps, frames, masks)):
                    raise ValueError("targets and masks must be lists")
                if not 0 <= anchor < length or not len(eps) == len(frames) == len(masks) > 0:
                    raise ValueError("invalid anchor or mismatched/empty lists")
                if any(type(x) is not int for x in eps + frames):
                    raise ValueError("target IDs must be integers")
                if any(type(x) is not bool for x in masks):
                    raise ValueError("padding_mask must contain booleans")
                for offset, (target_ep, frame, mask) in enumerate(
                    zip(eps, frames, masks, strict=True)
                ):
                    expected_frame = min(anchor + offset, length - 1)
                    expected_mask = anchor + offset >= length
                    if target_ep != episode:
                        report.add(
                            "WINDOW_CROSS_EPISODE",
                            "error",
                            "An action chunk contains a different episode.",
                            "Clamp to the current episode and mask padded targets.",
                            line=line_number,
                            offset=offset,
                            episode=episode,
                            target_episode=target_ep,
                        )
                    if frame != expected_frame or mask != expected_mask:
                        report.add(
                            "WINDOW_TARGET",
                            "error",
                            "Target frame or padding mask is wrong.",
                            "Compare the sampler to the documented contiguous trace convention.",
                            line=line_number,
                            offset=offset,
                            expected_frame=expected_frame,
                            actual_frame=frame,
                            expected_mask=expected_mask,
                            actual_mask=mask,
                        )
                count += 1
            except (ValueError, KeyError, TypeError) as exc:
                report.add(
                    "WINDOW_TRACE_INVALID",
                    "error",
                    "Sampler trace line is invalid.",
                    "Use the documented JSONL trace schema.",
                    line=line_number,
                    detail=str(exc),
                )
    if not count:
        report.add(
            "WINDOW_TRACE_EMPTY",
            "error",
            "No valid sampler trace rows were checked.",
            "Export samples from the actual training dataloader.",
        )
    report.coverage["sampler_trace_rows"] = count
