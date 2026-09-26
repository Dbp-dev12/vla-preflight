"""Streaming audits with explicit coverage. Source files are never changed."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pyarrow as pa

from .checks import check_contract, check_stats, check_windows
from .contract import Contract
from .dataset import Dataset, rows
from .report import Report


def _videos(ds, selected, mode, report):
    report.coverage["video_check"] = mode
    if mode == "skip":
        return
    ffprobe = shutil.which("ffprobe") if mode == "probe" else None
    if mode == "probe" and not ffprobe:
        report.complete = False
        report.add(
            "FFPROBE_UNAVAILABLE",
            "warning",
            "Requested video probing could not run.",
            "Install FFmpeg and add ffprobe to PATH, or request existence checks only.",
        )
        report.coverage["video_check"] = "exists only; ffprobe unavailable"
    cache = {}
    refs = 0
    try:
        for episode, key, path, end in ds.video_references(selected):
            refs += 1
            relative = path.relative_to(ds.root).as_posix()
            evidence = {"episode": episode, "feature": key, "file": relative}
            if not path.is_file():
                report.add(
                    "VIDEO_MISSING",
                    "error",
                    "Referenced video file is missing.",
                    "Download or restore this video; do not invent replacement frames.",
                    **evidence,
                )
                continue
            if ffprobe:
                if path not in cache:
                    try:
                        proc = subprocess.run(
                            [
                                ffprobe,
                                "-v",
                                "error",
                                "-select_streams",
                                "v:0",
                                "-show_entries",
                                "stream=codec_type,width,height:format=duration",
                                "-of",
                                "json",
                                str(path),
                            ],
                            capture_output=True,
                            text=True,
                            timeout=30,
                            check=False,
                        )
                        if proc.returncode:
                            raise ValueError("ffprobe rejected the file")
                        probe = json.loads(proc.stdout)
                        if not probe.get("streams"):
                            raise ValueError("no video stream")
                        cache[path] = probe
                    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
                        cache[path] = None
                        report.add(
                            "VIDEO_PROBE",
                            "error",
                            "Video container probe failed.",
                            "Check the video with ffprobe; this does not decode every frame.",
                            detail=str(exc),
                            **evidence,
                        )
                probe = cache[path]
                if probe and end is not None:
                    try:
                        duration = float(probe.get("format", {}).get("duration", "nan"))
                    except (TypeError, ValueError):
                        duration = float("nan")
                    if not np.isfinite(duration):
                        report.add(
                            "VIDEO_DURATION_UNKNOWN",
                            "warning",
                            "Video duration unavailable.",
                            "Inspect segment bounds manually.",
                            **evidence,
                        )
                    elif end > duration + 1 / ds.fps + 0.001:
                        report.add(
                            "VIDEO_SEGMENT_RANGE",
                            "error",
                            "Episode segment exceeds video duration.",
                            "Reconcile episode metadata with the correct video file.",
                            segment_end=end,
                            video_duration=duration,
                            **evidence,
                        )
    except (KeyError, ValueError, TypeError, ZeroDivisionError) as exc:
        report.complete = False
        report.add(
            "VIDEO_METADATA",
            "error",
            "Cannot resolve video references.",
            "Correct video paths and episode segment metadata.",
            detail=str(exc),
        )
    report.coverage["video_references_checked"] = refs
    report.coverage["video_files_probed"] = len(cache)


def audit(
    root: str | Path,
    *,
    contract: Contract | None = None,
    episodes: set[int] | None = None,
    video: str = "exists",
    windows: Path | None = None,
    timestamp_tolerance: float = 0.001,
) -> Report:
    if video not in ("exists", "skip", "probe"):
        raise ValueError("video must be exists, skip, or probe")
    if not np.isfinite(timestamp_tolerance) or timestamp_tolerance < 0:
        raise ValueError("timestamp_tolerance must be finite and nonnegative")
    report = Report()
    report.coverage.update(
        {
            "dataset": Path(root).name,
            "timestamp_tolerance_seconds": timestamp_tolerance,
            "training_contract": "provided" if contract else "not checked: no contract supplied",
            "sampler": "trace supplied"
            if windows
            else "not checked: no actual sampler trace supplied",
            "image_payloads": "not decoded",
            "physical_semantics": "not inferred",
        }
    )
    try:
        ds = Dataset(Path(root))
        selected = set(ds.episodes) if episodes is None else set(episodes)
        if not selected or not selected <= ds.episodes.keys():
            raise ValueError("Selected episodes must be a nonempty subset of metadata episode IDs")
        files = ds.data_files(selected)
    except (OSError, ValueError, TypeError, KeyError, pa.ArrowException) as exc:
        report.complete = False
        report.add(
            "DATASET_METADATA",
            "error",
            "Cannot resolve dataset metadata.",
            "Use a complete local LeRobot v2.1/v3.0 Parquet dataset, with meta/info.json.",
            detail=str(exc),
        )
        return report
    report.coverage.update(
        {
            "dataset_version": ds.version,
            "scope": "full" if selected == set(ds.episodes) else "selected episodes",
            "episodes_selected": len(selected),
            "episodes_in_metadata": len(ds.episodes),
            "selected_episode_ids": sorted(selected) if len(selected) <= 100 else "see input",
            "file_order": "metadata paths, lexical order; frame order preserved",
        }
    )
    for field, actual in (
        ("total_episodes", len(ds.episodes)),
        ("total_frames", sum(x["length"] for x in ds.episodes.values())),
    ):
        if ds.info.get(field) != actual:
            report.add(
                "METADATA_COUNT",
                "error",
                "Dataset counters disagree with episode metadata.",
                "Reconcile meta/info.json with episode lengths; check for a partial download.",
                field=field,
                declared=ds.info.get(field),
                actual=actual,
            )
    if contract:
        check_contract(ds, contract, report)
    check_stats(ds, contract, report)
    # Only numeric payloads: video/image features are intentionally excluded.
    numeric = {
        k: v
        for k, v in ds.features.items()
        if str(v.get("dtype", "")).startswith(("float", "int", "uint", "bool"))
    }
    for key in ("action", "observation.state"):
        if key in ds.features:
            numeric[key] = ds.features[key]
    columns = sorted(set(numeric) | {"episode_index", "frame_index", "timestamp", "index"})
    states = {ep: {"count": 0, "last_frame": -1, "last_time": None} for ep in selected}
    action_min, action_max = None, None
    for path in files:
        relative = path.relative_to(ds.root).as_posix()
        if not path.is_file():
            report.complete = False
            report.add(
                "DATA_FILE_MISSING",
                "error",
                "Referenced Parquet file is missing.",
                "Download or restore this file before training.",
                file=relative,
            )
            continue
        try:
            for row_number, row in enumerate(rows(path, columns)):
                ep = row.get("episode_index")
                location = {"file": relative, "row": row_number}
                if type(ep) is not int or ep not in ds.episodes:
                    report.add(
                        "EPISODE_ID",
                        "error",
                        "Invalid or unknown episode_index in data.",
                        "Match data rows to episode metadata.",
                        episode=ep,
                        **location,
                    )
                    continue
                if ep not in selected:
                    continue
                state = states[ep]
                state["count"] += 1
                report.coverage["rows_scanned"] += 1
                frame, time = row.get("frame_index"), row.get("timestamp")
                location.update(episode=ep, frame=frame)
                if type(frame) is not int or frame != state["last_frame"] + 1:
                    report.add(
                        "FRAME_SEQUENCE",
                        "error",
                        "Frames are missing, duplicated or out of order.",
                        "Preserve frame order and zero-based consecutive indices per episode.",
                        previous_frame=state["last_frame"],
                        **location,
                    )
                if type(frame) is int:
                    state["last_frame"] = frame
                    index = row.get("index")
                    expected_index = ds.offsets[ep] + frame
                    if type(index) is not int or index != expected_index:
                        report.add(
                            "GLOBAL_INDEX",
                            "error",
                            "Global index disagrees with episode/frame.",
                            "Preserve global indices and episode-relative frame indices.",
                            expected_index=expected_index,
                            actual_index=index,
                            **location,
                        )
                valid_time = (
                    isinstance(time, (int, float))
                    and not isinstance(time, bool)
                    and np.isfinite(time)
                )
                if not valid_time:
                    report.add(
                        "TIMESTAMP_INVALID",
                        "error",
                        "Timestamp is missing or nonfinite.",
                        "Use finite episode-relative timestamps in seconds.",
                        **location,
                    )
                else:
                    if state["last_time"] is not None and time <= state["last_time"]:
                        report.add(
                            "TIMESTAMP_ORDER",
                            "error",
                            "Timestamps are not strictly increasing.",
                            "Check duplicate or reversed observations.",
                            **location,
                        )
                    if type(frame) is int and abs(time - frame / ds.fps) > timestamp_tolerance:
                        report.add(
                            "TIMESTAMP_FPS",
                            "error",
                            "Timestamp differs from frame_index/fps.",
                            "Check time units and recording frequency. Adjust tolerance only "
                            "for a documented timestamp policy.",
                            observed=time,
                            expected=frame / ds.fps,
                            **location,
                        )
                    state["last_time"] = time
                for key, spec in numeric.items():
                    if key not in row or row[key] is None:
                        report.add(
                            "FEATURE_MISSING",
                            "error",
                            "Numeric feature is missing or null.",
                            "Restore the declared column and values.",
                            feature=key,
                            **location,
                        )
                        continue
                    try:
                        if np.asarray(row[key]).dtype.kind not in "biuf":
                            raise ValueError(
                                "Expected a numeric Parquet value, not a numeric string"
                            )
                        arr = np.asarray(row[key], dtype=float)
                        # Scalar index/time fields use shape [1] in LeRobot metadata.
                        actual_shape = list(arr.shape) if arr.ndim else [1]
                        if actual_shape != spec["shape"]:
                            report.add(
                                "FEATURE_SHAPE",
                                "error",
                                "Feature shape disagrees with metadata.",
                                "Fix the conversion or feature schema.",
                                feature=key,
                                expected=spec["shape"],
                                actual=actual_shape,
                                **location,
                            )
                            continue
                        if not np.isfinite(arr).all():
                            report.add(
                                "NONFINITE",
                                "error",
                                "Numeric feature contains NaN or infinity.",
                                "Inspect the source episode; do not silently replace invalid data.",
                                feature=key,
                                **location,
                            )
                            continue
                        if key == "action":
                            action_min = (
                                arr.copy() if action_min is None else np.minimum(action_min, arr)
                            )
                            action_max = (
                                arr.copy() if action_max is None else np.maximum(action_max, arr)
                            )
                    except (TypeError, ValueError):
                        report.add(
                            "FEATURE_TYPE",
                            "error",
                            "Numeric feature cannot be read as numbers.",
                            "Fix the Parquet column type.",
                            feature=key,
                            **location,
                        )
            report.coverage["files_scanned"] += 1
        except (OSError, ValueError, pa.ArrowException) as exc:
            report.complete = False
            report.add(
                "PARQUET_READ",
                "error",
                "Parquet file could not be fully read.",
                "Restore or regenerate the file.",
                file=relative,
                detail=str(exc),
            )
    for ep, state in states.items():
        expected = ds.episodes[ep]["length"]
        if state["count"] != expected:
            report.add(
                "EPISODE_LENGTH",
                "error",
                "Scanned episode length differs from metadata.",
                "Check missing files, wrong episode IDs, or truncated exports.",
                episode=ep,
                expected=expected,
                actual=state["count"],
            )
    if action_min is not None and np.any(action_min == action_max):
        report.add(
            "CONSTANT_ACTION",
            "warning",
            "Some action dimensions are constant in scanned data.",
            "Check inactive/fixed joints; constancy alone is not a defect.",
            dimensions=np.flatnonzero(action_min == action_max).tolist(),
        )
    _videos(ds, selected, video, report)
    if windows:
        try:
            check_windows(windows, ds, report)
        except OSError as exc:
            report.complete = False
            report.add(
                "WINDOW_TRACE_READ",
                "error",
                "Cannot read sampler trace.",
                "Supply a readable JSONL trace.",
                detail=str(exc),
            )
    return report
