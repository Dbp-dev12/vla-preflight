"""Honest task-level summaries for simulation or supervised robot rollouts."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt

from . import __version__
from .workflow_io import atomic_json


class Rollout(BaseModel):
    model_config = ConfigDict(extra="forbid")
    episode_id: str = Field(min_length=1, max_length=160)
    success: StrictBool
    duration_s: float = Field(gt=0, le=86400, allow_inf_nan=False)
    interventions: StrictInt = Field(default=0, ge=0, le=100000)
    task: str = Field(default="default", min_length=1, max_length=500)
    failure_mode: str | None = Field(default=None, max_length=300)
    checkpoint: str | None = Field(default=None, max_length=500)
    dataset_digest: str | None = Field(default=None, max_length=128)


def _interval(successes: int, total: int):
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, centre - radius), min(1.0, centre + radius)]


def _group(records):
    successes = sum(item.success for item in records)
    return {
        "episodes": len(records),
        "successes": successes,
        "success_rate": successes / len(records),
        "wilson_95_interval": _interval(successes, len(records)),
        "mean_duration_s": sum(item.duration_s for item in records) / len(records),
        "total_interventions": sum(item.interventions for item in records),
        "episodes_with_intervention": sum(item.interventions > 0 for item in records),
    }


def evaluate_rollouts(source: Path):
    records = []
    seen = set()
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = Rollout.model_validate(json.loads(line))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"Invalid rollout JSONL at line {line_number}: {exc}") from exc
            if item.episode_id in seen:
                raise ValueError(f"Duplicate rollout episode_id: {item.episode_id}")
            seen.add(item.episode_id)
            records.append(item)
            if len(records) > 100000:
                raise ValueError("Rollout log exceeds 100,000 records")
    if not records:
        raise ValueError("Rollout log is empty")
    tasks = defaultdict(list)
    checkpoints = defaultdict(list)
    for item in records:
        tasks[item.task].append(item)
        checkpoints[item.checkpoint or "unspecified"].append(item)
    failures = Counter(item.failure_mode or "unspecified" for item in records if not item.success)
    return {
        "schema": "vla-preflight.rollout-evaluation/1",
        "tool_version": __version__,
        "overall": _group(records),
        "by_task": {name: _group(items) for name, items in sorted(tasks.items())},
        "by_checkpoint": {name: _group(items) for name, items in sorted(checkpoints.items())},
        "failure_modes": dict(failures.most_common()),
        "dataset_digests": sorted({x.dataset_digest for x in records if x.dataset_digest}),
        "limits": [
            "The Wilson interval describes binomial sampling uncertainty, not deployment safety.",
            "Unblinded operators, task selection and interventions can bias success estimates.",
            "Offline action error and training loss are not substituted for task success.",
        ],
    }


def write_rollout_evaluation(source: Path, destination: Path):
    if source.resolve() == destination.resolve():
        raise ValueError("Rollout report must not overwrite its source log")
    report = evaluate_rollouts(source)
    atomic_json(destination, report)
    return report
