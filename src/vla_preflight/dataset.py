"""Small LeRobot layout reader. Parquet batches, not decoded images, enter memory."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from .contract import load_json


def inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Dataset path escapes root: {relative}")
    return path


def rows(path: Path, columns: list[str] | None = None):
    # Do not materialize image byte columns when auditing numeric fields.
    parquet = pq.ParquetFile(path)
    if columns is not None:
        columns = [c for c in columns if c in parquet.schema_arrow.names]
    for batch in parquet.iter_batches(batch_size=2048, columns=columns):
        yield from batch.to_pylist()


class Dataset:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.info = load_json(inside(self.root, "meta/info.json"))
        if not isinstance(self.info, dict):
            raise ValueError("meta/info.json must contain an object")
        self.version = self.info.get("codebase_version")
        if self.version not in ("v2.1", "v3.0"):
            raise ValueError(f"Unsupported codebase_version: {self.version!r}; use v2.1 or v3.0")
        if self.info.get("storage_format") not in (None, "parquet"):
            raise ValueError("Only the Parquet storage backend is supported")
        self.features = self.info.get("features")
        if not isinstance(self.features, dict) or "action" not in self.features:
            raise ValueError("features must include an action definition")
        for key, spec in self.features.items():
            if not isinstance(spec, dict) or not isinstance(spec.get("shape"), list):
                raise ValueError(f"Invalid feature shape: {key}")
            if any(type(d) is not int or d <= 0 for d in spec["shape"]):
                raise ValueError(f"Shape must contain positive integer dimensions: {key}")
        for key in ("action", "observation.state"):
            if key in self.features and len(self.features[key]["shape"]) != 1:
                raise ValueError(f"Only vector features are supported for {key}")
        for key in ("data_path", "video_path"):
            if self.info.get(key) is not None and not isinstance(self.info[key], str):
                raise ValueError(f"{key} must be a path template string")
        fps = self.info.get("fps")
        if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not 0 < fps < 1e6:
            raise ValueError("fps must be a finite positive number below 1e6")
        self.fps = float(fps)
        self.episodes: dict[int, dict] = {}
        if self.version == "v2.1":
            with inside(self.root, "meta/episodes.jsonl").open(encoding="utf-8-sig") as f:
                for line in f:
                    if line.strip():
                        self._add_episode(json.loads(line))
        else:
            paths = sorted(inside(self.root, "meta/episodes").rglob("*.parquet"))
            if not paths:
                raise ValueError("v3.0 requires meta/episodes/**/*.parquet")
            for path in paths:
                for row in rows(inside(self.root, str(path.relative_to(self.root)))):
                    self._add_episode(row)
        if not self.episodes:
            raise ValueError("Dataset has no episode metadata")
        self.offsets = {}
        offset = 0
        for episode, meta in sorted(self.episodes.items()):
            self.offsets[episode] = offset
            if self.version == "v3.0":
                start, end = meta.get("dataset_from_index"), meta.get("dataset_to_index")
                if type(start) is not int or type(end) is not int:
                    raise ValueError("v3.0 episode metadata needs integer dataset_from/to_index")
                if start != offset or end != start + meta["length"]:
                    raise ValueError(f"Inconsistent v3.0 global index range for episode {episode}")
            offset += meta["length"]

    def _add_episode(self, row):
        if not isinstance(row, dict):
            raise ValueError("Episode metadata must be an object")
        index, length = row.get("episode_index"), row.get("length")
        if type(index) is not int or index < 0 or type(length) is not int or length <= 0:
            raise ValueError("Episode metadata needs nonnegative episode_index and positive length")
        if index in self.episodes:
            raise ValueError(f"Duplicate episode metadata: {index}")
        self.episodes[index] = row

    def data_files(self, selected: set[int]):
        paths = set()
        for episode in selected:
            meta = self.episodes[episode]
            if self.version == "v2.1":
                size = self.info.get("chunks_size", 1000)
                if type(size) is not int or size <= 0:
                    raise ValueError("chunks_size must be a positive integer")
                template = self.info.get(
                    "data_path",
                    "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                )
                relative = template.format(episode_chunk=episode // size, episode_index=episode)
            else:
                template = self.info.get(
                    "data_path", "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
                )
                relative = template.format(
                    chunk_index=meta["data/chunk_index"], file_index=meta["data/file_index"]
                )
            paths.add(inside(self.root, relative))
        return sorted(paths)

    def video_references(self, selected: set[int]):
        for episode in sorted(selected):
            meta = self.episodes[episode]
            for key, spec in self.features.items():
                if spec.get("dtype") != "video":
                    continue
                template = self.info.get("video_path")
                if not template:
                    raise ValueError("Video features declared but video_path is missing")
                end = None
                if self.version == "v2.1":
                    relative = template.format(
                        video_key=key,
                        episode_chunk=episode // self.info.get("chunks_size", 1000),
                        episode_index=episode,
                    )
                else:
                    prefix = f"videos/{key}"
                    relative = template.format(
                        video_key=key,
                        chunk_index=meta[f"{prefix}/chunk_index"],
                        file_index=meta[f"{prefix}/file_index"],
                    )
                    start, end = (
                        meta.get(f"{prefix}/from_timestamp"),
                        meta.get(f"{prefix}/to_timestamp"),
                    )
                    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                        raise ValueError(f"Missing video segment bounds: episode {episode}, {key}")
                    if not 0 <= start < end < float("inf"):
                        raise ValueError(f"Invalid video segment bounds: episode {episode}, {key}")
                yield episode, key, inside(self.root, relative), end
