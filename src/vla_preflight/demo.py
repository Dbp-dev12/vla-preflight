"""Deterministic, generated fixtures, not robot demonstrations or performance claims."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

FAULTS = {
    "nan-action": "NONFINITE",
    "wrong-dimension": "CONFIG_DIMENSION",
    "missing-camera": "CONFIG_CAMERA",
    "split-overlap": "SPLIT_OVERLAP",
    "timestamp-shift": "TIMESTAMP_FPS",
    "negative-std": "STATS_NEGATIVE_STD",
    "joint-order": "SEMANTICS_MISMATCH",
    "cross-episode": "WINDOW_CROSS_EPISODE",
    "missing-shard": "DATA_FILE_MISSING",
}


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def create_demo(root: Path, *, version="v3.0", fault="none") -> Path:
    if root.exists():
        raise ValueError(f"Destination already exists; choose a new directory: {root}")
    if version not in ("v2.1", "v3.0") or fault not in ("none", *FAULTS):
        raise ValueError("Unknown demo version or fault")
    root.mkdir(parents=True)
    data = root / "dataset"
    features = {
        "action": {"dtype": "float32", "shape": [2], "names": ["joint_a", "joint_b"]},
        "observation.state": {"dtype": "float32", "shape": [2], "names": ["joint_a", "joint_b"]},
        **{
            key: {"dtype": "float32" if key == "timestamp" else "int64", "shape": [1]}
            for key in ("timestamp", "frame_index", "episode_index", "index", "task_index")
        },
    }
    info = {
        "codebase_version": version,
        "fps": 10,
        "total_episodes": 2,
        "total_frames": 12,
        "total_tasks": 1,
        "chunks_size": 1000,
        "robot_type": "synthetic_fixture",
        "features": features,
        "video_path": None,
    }
    info["data_path"] = (
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        if version == "v2.1"
        else "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    )
    stats = {
        key: {"mean": [0.5, 1.5], "std": [0.3, 0.3], "min": [0, 1], "max": [1, 2]}
        for key in ("action", "observation.state")
    }
    semantics = {
        "action_names": ["joint_a", "joint_b"],
        "action_units": ["rad", "rad"],
        "action_mode": "absolute",
        "coordinate_frame": "joint_space",
    }
    contract = {
        "schema_version": 1,
        "action_dim": 2,
        "state_dim": 2,
        "fps": 10,
        "normalization": {"action": "MEAN_STD", "observation.state": "MEAN_STD"},
        "train_episodes": [0],
        "validation_episodes": [1],
        "training": semantics.copy(),
        "deployment": semantics.copy(),
    }
    traces = [
        {
            "episode_index": 0,
            "anchor_frame": 4,
            "target_episode_indices": [0, 0, 0, 0],
            "target_frame_indices": [4, 5, 5, 5],
            "padding_mask": [False, False, True, True],
        }
    ]
    if fault == "wrong-dimension":
        contract["state_dim"] = 7
    elif fault == "missing-camera":
        contract["camera_keys"] = ["observation.images.wrist"]
    elif fault == "split-overlap":
        contract["validation_episodes"] = [0, 1]
    elif fault == "negative-std":
        stats["action"]["std"][0] = -0.3
    elif fault == "joint-order":
        contract["deployment"]["action_names"] = ["joint_b", "joint_a"]
    elif fault == "cross-episode":
        traces[0]["target_episode_indices"][2:] = [1, 1]
        traces[0]["target_frame_indices"][2:] = [0, 1]
        traces[0]["padding_mask"][2:] = [False, False]
    records = []
    for ep in range(2):
        for frame in range(6):
            x = (ep * 6 + frame) / 11
            records.append(
                {
                    "action": [x, 1 + x],
                    "observation.state": [x, 1 + x],
                    "timestamp": frame / 10,
                    "episode_index": ep,
                    "frame_index": frame,
                    "index": ep * 6 + frame,
                    "task_index": 0,
                }
            )
    if fault == "nan-action":
        records[3]["action"][0] = float("nan")
    elif fault == "timestamp-shift":
        records[3]["timestamp"] += 0.04
    schema = pa.schema(
        [
            pa.field("action", pa.list_(pa.float32())),
            pa.field("observation.state", pa.list_(pa.float32())),
            pa.field("timestamp", pa.float32()),
            *[
                pa.field(key, pa.int64())
                for key in ("episode_index", "frame_index", "index", "task_index")
            ],
        ]
    )
    episodes = [
        {"episode_index": ep, "length": 6, "tasks": ["synthetic motion"]} for ep in range(2)
    ]
    write_json(data / "meta/info.json", info)
    write_json(data / "meta/stats.json", stats)
    if version == "v2.1":
        (data / "meta/episodes.jsonl").write_text(
            "".join(json.dumps(x) + "\n" for x in episodes), encoding="utf-8"
        )
        (data / "meta/tasks.jsonl").write_text(
            json.dumps({"task_index": 0, "task": "synthetic motion"}) + "\n", encoding="utf-8"
        )
        for ep in range(2):
            path = data / f"data/chunk-000/episode_{ep:06d}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            if fault != "missing-shard" or ep == 0:
                pq.write_table(
                    pa.Table.from_pylist(records[ep * 6 : (ep + 1) * 6], schema=schema), path
                )
    else:
        for ep in episodes:
            ep.update(
                {
                    "data/chunk_index": 0,
                    "data/file_index": 0,
                    "dataset_from_index": ep["episode_index"] * 6,
                    "dataset_to_index": (ep["episode_index"] + 1) * 6,
                }
            )
        path = data / "meta/episodes/chunk-000/file-000.parquet"
        path.parent.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(episodes), path)
        pq.write_table(
            pa.Table.from_pylist([{"task_index": 0, "task": "synthetic motion"}]),
            data / "meta/tasks.parquet",
        )
        path = data / "data/chunk-000/file-000.parquet"
        path.parent.mkdir(parents=True)
        if fault != "missing-shard":
            pq.write_table(pa.Table.from_pylist(records, schema=schema), path)
    write_json(root / "contract.json", contract)
    (root / "windows.jsonl").write_text(
        "".join(json.dumps(x) + "\n" for x in traces), encoding="utf-8"
    )
    write_json(
        root / "provenance.json",
        {
            "source": "generated synthetic numeric fixture",
            "seed": "deterministic formula; no RNG",
            "fault": fault,
            "expected_code": FAULTS.get(fault),
            "not_robot_data": True,
        },
    )
    return root
