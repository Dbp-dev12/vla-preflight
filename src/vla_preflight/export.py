"""Explicit episode filtering into a new v3 dataset, preserving source data."""

from __future__ import annotations

import copy
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .analysis import train_statistics
from .audit import audit
from .dataset import Dataset, inside
from .workflow_io import atomic_json, episode_records, fingerprint, new_artifact_dir, tasks


def export_episodes(root: Path, destination: Path, *, exclude: set[int]):
    ds = Dataset(root)
    if not exclude <= ds.episodes.keys():
        raise ValueError("Unknown excluded episode IDs")
    selected = sorted(ds.episodes.keys() - exclude)
    if not selected:
        raise ValueError("Cannot export an empty dataset")
    check = audit(root, episodes=set(selected))
    if not check.complete or check.status == "failed":
        raise ValueError("Selected episodes have preflight errors; cannot export a valid dataset")
    source = fingerprint(ds, videos=True)
    # Resolve media before creating output. Copy shared videos once without recompression.
    references = list(ds.video_references(set(selected)))
    media_map = {}
    for _, key, path, _ in references:
        if (key, path) not in media_map:
            media_map[key, path] = len(media_map)
    destination = new_artifact_dir(destination, root)
    atomic_json(destination / "export-status.json", {"complete": False})
    info = copy.deepcopy(ds.info)
    info.update(
        {
            "codebase_version": "v3.0",
            "total_episodes": len(selected),
            "total_frames": sum(ds.episodes[ep]["length"] for ep in selected),
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "splits": {"train": f"0:{len(selected)}"},
        }
    )
    metadata, offset = [], 0
    for new_ep, old_ep in enumerate(selected):
        records = episode_records(ds, old_ep)
        for frame, row in enumerate(records):
            row.update(episode_index=new_ep, frame_index=frame, index=offset + frame)
            for key, spec in ds.features.items():
                value = row.get(key)
                if spec.get("dtype") == "image" and isinstance(value, dict):
                    if value.get("bytes") is None and value.get("path"):
                        # Materialize referenced images so export is independently readable.
                        row[key] = {
                            "bytes": inside(ds.root, value["path"]).read_bytes(),
                            "path": None,
                        }
        data_path = (
            destination / f"data/chunk-{new_ep // 1000:03d}/file-{new_ep % 1000:03d}.parquet"
        )
        data_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(records), data_path)
        meta = {
            "episode_index": new_ep,
            "length": len(records),
            "tasks": ds.episodes[old_ep].get("tasks", []),
            "data/chunk_index": new_ep // 1000,
            "data/file_index": new_ep % 1000,
            "dataset_from_index": offset,
            "dataset_to_index": offset + len(records),
        }
        for ep, key, path, end in references:
            if ep != old_ep:
                continue
            number = media_map[key, path]
            prefix = f"videos/{key}"
            start = 0.0 if ds.version == "v2.1" else ds.episodes[old_ep][f"{prefix}/from_timestamp"]
            stop = len(records) / ds.fps if ds.version == "v2.1" else end
            meta.update(
                {
                    f"{prefix}/chunk_index": number // 1000,
                    f"{prefix}/file_index": number % 1000,
                    f"{prefix}/from_timestamp": start,
                    f"{prefix}/to_timestamp": stop,
                }
            )
        metadata.append(meta)
        offset += len(records)
    for (key, source_path), number in media_map.items():
        target = (
            destination / f"videos/{key}/chunk-{number // 1000:03d}/file-{number % 1000:03d}.mp4"
        )
        target = inside(destination, str(target.relative_to(destination)))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target)
    meta_path = destination / "meta/episodes/chunk-000/file-000.parquet"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(metadata), meta_path)
    task_map = tasks(ds)
    if not task_map:
        raise ValueError("Task metadata is required for standalone export")
    pq.write_table(
        pa.Table.from_pylist([{"task_index": k, "task": v} for k, v in task_map.items()]),
        destination / "meta/tasks.parquet",
    )
    info["total_tasks"] = len(task_map)
    atomic_json(destination / "meta/info.json", info)
    stats = train_statistics(Dataset(destination), set(range(len(selected))))
    lerobot_stats = {
        key: {k: v for k, v in values.items() if k in ("mean", "std", "min", "max", "q01", "q99")}
        for key, values in stats.items()
    }
    for key, values in lerobot_stats.items():
        values["count"] = [stats[key]["count"]]
    atomic_json(destination / "meta/stats.json", lerobot_stats)
    verification = audit(destination)
    if not verification.complete or verification.status == "failed":
        raise ValueError("Export failed verification; export-status.json remains incomplete")
    provenance = {
        "complete": True,
        "source_digest": source["digest"],
        "episode_mapping": {str(old): new for new, old in enumerate(selected)},
        "excluded_episodes": sorted(exclude),
        "transformations": ["episode/global-index reindexing"],
        "statistics": "recomputed for exported action/state; visual stats not supplied",
        "quantiles": {k: v["quantile_method"] for k, v in stats.items()},
        "video": "referenced files copied unchanged; shared files may retain excluded footage",
    }
    atomic_json(destination / "export-status.json", provenance)
    return provenance
