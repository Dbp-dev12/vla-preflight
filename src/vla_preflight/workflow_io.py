"""Shared local artifact and dataset access. No automatic network access."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .contract import load_json
from .dataset import Dataset, inside, rows


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def new_artifact_dir(path: Path, source: Path):
    path = path.resolve()
    if path.is_relative_to(source.resolve()):
        raise ValueError("Output must be outside the source dataset")
    if path.exists():
        raise ValueError(f"Output already exists; choose a new directory: {path}")
    path.mkdir(parents=True)
    return path


def file_hash(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_json(obj):
    return hashlib.sha256(
        json.dumps(
            obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


def fingerprint(ds: Dataset, *, videos=False):
    paths = set(ds.data_files(set(ds.episodes)))
    image_keys = [k for k, feature in ds.features.items() if feature.get("dtype") == "image"]
    if image_keys:
        for parquet in list(paths):
            for row in rows(parquet, image_keys):
                for key in image_keys:
                    image = row.get(key)
                    if isinstance(image, dict) and not image.get("bytes") and image.get("path"):
                        paths.add(inside(ds.root, image["path"]))
    paths.update(p for p in inside(ds.root, "meta").rglob("*") if p.is_file())
    if videos:
        paths.update(path for _, _, path, _ in ds.video_references(set(ds.episodes)))
    files = {}
    for path in sorted(paths):
        path = inside(ds.root, str(path.relative_to(ds.root)))
        if not path.is_file():
            raise ValueError(f"Missing fingerprint input: {path.name}")
        relative = path.relative_to(ds.root).as_posix()
        files[relative] = {"size": path.stat().st_size, "sha256": file_hash(path)}
    return {
        "algorithm": "sha256",
        "digest": digest_json(files),
        "files": files,
        "video_content_hashed": videos,
        "scope": "metadata, referenced Parquet and image paths; embedded image bytes included",
    }


def episode_records(ds: Dataset, episode: int, columns=None):
    if episode not in ds.episodes:
        raise ValueError(f"Unknown episode: {episode}")
    return [
        row
        for path in ds.data_files({episode})
        for row in rows(path, columns)
        if row.get("episode_index") == episode
    ]


def tasks(ds: Dataset):
    result = {}
    if ds.version == "v2.1":
        path = inside(ds.root, "meta/tasks.jsonl")
        if path.is_file():
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                if line.strip():
                    item = json.loads(line)
                    result[int(item["task_index"])] = str(item["task"])
    else:
        path = inside(ds.root, "meta/tasks.parquet")
        if path.is_file():
            for item in rows(path):
                # Pandas-backed LeRobot v3 uses task strings as the index.
                text = item.get("task", item.get("__index_level_0__"))
                if text is not None:
                    result[int(item["task_index"])] = str(text)
    return result


def load_prepared(path: Path, *, verify=True):
    manifest = load_json(path / "manifest.json")
    if manifest.get("schema") != "vla-preflight.prepare/1":
        raise ValueError("Unsupported prepared bundle schema")
    if not manifest.get("complete"):
        raise ValueError("Preparation did not complete")
    ds = Dataset(Path(manifest["dataset_root"]))
    if verify:
        actual = fingerprint(ds, videos=manifest["source"]["video_content_hashed"])
        if actual["digest"] != manifest["source"]["digest"]:
            raise ValueError("Source changed after preparation; create a new prepared bundle")
        for name, expected in manifest["artifact_hashes"].items():
            if file_hash(inside(path, name)) != expected:
                raise ValueError(f"Prepared artifact changed: {name}; create a new bundle")
    return ds, manifest, load_json(path / "split.json"), load_json(path / "normalization.json")
