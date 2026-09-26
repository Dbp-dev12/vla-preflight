"""A toy visual-language reaching dataset, generated locally; not real robot data."""

from __future__ import annotations

import io

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .analysis import train_statistics
from .dataset import Dataset
from .workflow_io import atomic_json


def learning_demo(destination, *, seed=7, episodes=24, frames=16):
    from PIL import Image, ImageDraw

    if destination.exists():
        raise ValueError("Demo destination already exists")
    if not 4 <= episodes <= 1000 or not 4 <= frames <= 200:
        raise ValueError("Use 4–1000 episodes and 4–200 frames")
    rng = np.random.default_rng(seed)
    records, metadata = [], []
    for ep in range(episodes):
        red, blue, state = rng.uniform(0.15, 0.85, size=(3, 2))
        label = ep % 2
        target = red if label == 0 else blue
        for frame in range(frames):
            image = Image.new("RGB", (32, 32), (20, 27, 43))
            draw = ImageDraw.Draw(image)
            for xy, color in (
                (red, (240, 75, 70)),
                (blue, (60, 130, 245)),
                (state, (245, 245, 245)),
            ):
                x, y = (xy * 31).astype(int)
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            action = (target - state) * 0.18
            records.append(
                {
                    "observation.state": state.tolist(),
                    "action": action.tolist(),
                    "observation.image": {"bytes": buffer.getvalue(), "path": None},
                    "timestamp": frame / 10,
                    "episode_index": ep,
                    "frame_index": frame,
                    "index": ep * frames + frame,
                    "task_index": label,
                }
            )
            state = state + action
        metadata.append(
            {
                "episode_index": ep,
                "length": frames,
                "tasks": ["reach red target" if label == 0 else "reach blue target"],
                "data/chunk_index": 0,
                "data/file_index": 0,
                "dataset_from_index": ep * frames,
                "dataset_to_index": (ep + 1) * frames,
            }
        )
    features = {k: {"dtype": "float32", "shape": [2]} for k in ("action", "observation.state")}
    features.update(
        {
            "observation.image": {"dtype": "image", "shape": [32, 32, 3]},
            **{
                k: {"dtype": "float32" if k == "timestamp" else "int64", "shape": [1]}
                for k in ("timestamp", "episode_index", "frame_index", "index", "task_index")
            },
        }
    )
    info = {
        "codebase_version": "v3.0",
        "fps": 10,
        "total_episodes": episodes,
        "total_frames": len(records),
        "total_tasks": 2,
        "features": features,
        "robot_type": "synthetic_color_reaching",
        "video_path": None,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
    }
    for relative, content in (
        ("data/chunk-000/file-000.parquet", records),
        ("meta/episodes/chunk-000/file-000.parquet", metadata),
        (
            "meta/tasks.parquet",
            [
                {"task_index": 0, "task": "reach red target"},
                {"task_index": 1, "task": "reach blue target"},
            ],
        ),
    ):
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(content), path)
    atomic_json(destination / "meta/info.json", info)
    statistics = train_statistics(Dataset(destination), set(range(episodes)))
    atomic_json(
        destination / "meta/stats.json",
        {
            k: {
                name: value
                for name, value in stats.items()
                if name in ("mean", "std", "min", "max")
            }
            for k, stats in statistics.items()
        },
    )
    atomic_json(
        destination / "DEMO.json",
        {
            "synthetic": True,
            "seed": seed,
            "purpose": "Validate training plumbing, not robot task performance",
        },
    )
    return destination
