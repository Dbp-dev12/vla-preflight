"""Honest task-level summaries for simulation or supervised robot rollouts."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from html import escape
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, field_validator

from . import __version__
from .contract import load_json
from .workflow_io import atomic_json, digest_json, file_hash


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


class EvaluationProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid")
    protocol_id: str = Field(min_length=1, max_length=120, pattern=r"^[A-Za-z0-9_.-]+$")
    task: str = Field(min_length=1, max_length=500)
    success_definition: str = Field(min_length=10, max_length=2000)
    reset_procedure: str = Field(min_length=10, max_length=2000)
    environment: str = Field(min_length=1, max_length=500)
    required_episodes: StrictInt = Field(ge=1, le=100000)
    max_duration_s: float = Field(gt=0, le=86400, allow_inf_nan=False)
    allowed_failure_modes: list[str] = Field(default_factory=list, max_length=100)

    @field_validator("allowed_failure_modes")
    @classmethod
    def unique_modes(cls, value):
        if len(set(value)) != len(value):
            raise ValueError("failure modes must be unique")
        return value


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


def _load_rollouts(source: Path):
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
    return records


def _evaluate(records):
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


def evaluate_rollouts(source: Path):
    return _evaluate(_load_rollouts(source))


def write_rollout_evaluation(source: Path, destination: Path):
    if source.resolve() == destination.resolve():
        raise ValueError("Rollout report must not overwrite its source log")
    report = evaluate_rollouts(source)
    atomic_json(destination, report)
    return report


def create_rollout_session(
    source: Path,
    protocol_path: Path,
    destination: Path,
    *,
    robot_plan: Path,
    checkpoint: Path,
    dataset_digest: str,
):
    """Bind outcomes to the exact evaluation protocol, robot, data and policy."""
    if not dataset_digest or len(dataset_digest) != 64:
        raise ValueError("dataset_digest must be a 64-character SHA-256 digest")
    try:
        int(dataset_digest, 16)
    except ValueError as exc:
        raise ValueError("dataset_digest must be hexadecimal") from exc
    protocol = EvaluationProtocol.model_validate(load_json(protocol_path))
    plan = load_json(robot_plan)
    if plan.get("schema") != "vla-preflight.robot-plan/1" or not plan.get("safe_to_start"):
        raise ValueError("A ready vla-preflight robot plan is required")
    if not checkpoint.is_file():
        raise ValueError("Checkpoint file is missing")
    records = _load_rollouts(source)
    for item in records:
        if item.task != protocol.task:
            raise ValueError(f"Rollout {item.episode_id} does not match the protocol task")
        if item.duration_s > protocol.max_duration_s:
            raise ValueError(f"Rollout {item.episode_id} exceeds protocol max_duration_s")
        if item.failure_mode and protocol.allowed_failure_modes:
            if item.failure_mode not in protocol.allowed_failure_modes:
                raise ValueError(f"Rollout {item.episode_id} uses an undeclared failure mode")
    destination = destination.resolve()
    if destination.exists():
        raise ValueError(f"Output already exists; choose a new directory: {destination}")
    destination.mkdir(parents=True)
    summary = _evaluate(records)
    protocol_data = protocol.model_dump(mode="json")
    identity = {
        "protocol_digest": digest_json(protocol_data),
        "robot_plan_digest": digest_json(plan),
        "dataset_digest": dataset_digest.lower(),
        "checkpoint": {
            "name": checkpoint.name,
            "size": checkpoint.stat().st_size,
            "sha256": file_hash(checkpoint),
        },
    }
    normalized = destination / "episodes.jsonl"
    normalized.write_text(
        "".join(json.dumps(item.model_dump(), ensure_ascii=False) + "\n" for item in records),
        encoding="utf-8",
    )
    complete = len(records) >= protocol.required_episodes
    manifest = {
        "schema": "vla-preflight.rollout-session/1",
        "tool_version": __version__,
        "status": "complete" if complete else "incomplete",
        "complete": complete,
        "identity": identity,
        "protocol": protocol_data,
        "episodes": len(records),
        "required_episodes": protocol.required_episodes,
        "artifacts": {
            "episodes.jsonl": file_hash(normalized),
            "source_log_sha256": file_hash(source),
        },
        "summary": summary,
        "limits": summary["limits"],
    }
    atomic_json(destination / "summary.json", summary)
    atomic_json(destination / "manifest.json", manifest)
    low, high = summary["overall"]["wilson_95_interval"]
    failures = escape(json.dumps(summary["failure_modes"], ensure_ascii=False, indent=2))
    bound_identity = escape(json.dumps(identity, ensure_ascii=False, indent=2))
    html = f"""<!doctype html><meta charset=\"utf-8\"><title>Rollout session</title>
<h1>{escape(protocol.protocol_id)}</h1>
<p><b>Status:</b> {manifest["status"]} · {len(records)}/{protocol.required_episodes} episodes</p>
<p><b>Task:</b> {escape(protocol.task)}</p>
<p><b>Success:</b> {summary["overall"]["successes"]}/{len(records)}
({summary["overall"]["success_rate"]:.1%})</p>
<p><b>95% Wilson interval:</b> {low:.1%}–{high:.1%}</p>
<h2>Failure modes</h2><pre>{failures}</pre>
<h2>Bound identity</h2><pre>{bound_identity}</pre>
<p>Observed task outcomes are not a deployment-safety certificate.</p>"""
    (destination / "report.html").write_text(html, encoding="utf-8")
    return manifest


def compare_rollout_sessions(paths: list[Path]):
    if len(paths) < 2:
        raise ValueError("Provide at least two rollout session directories")
    sessions = [load_json(path / "manifest.json") for path in paths]
    if any(item.get("schema") != "vla-preflight.rollout-session/1" for item in sessions):
        raise ValueError("All inputs must be rollout sessions")
    protocol = {item["identity"]["protocol_digest"] for item in sessions}
    robot = {item["identity"]["robot_plan_digest"] for item in sessions}
    dataset = {item["identity"]["dataset_digest"] for item in sessions}
    comparable = len(protocol) == len(robot) == len(dataset) == 1
    return {
        "schema": "vla-preflight.rollout-comparison/1",
        "comparable": comparable,
        "reason": (
            "Same protocol, robot plan and dataset"
            if comparable
            else "Protocol, robot plan or dataset differs; rates are not directly comparable"
        ),
        "sessions": [
            {
                "name": path.name,
                "status": item["status"],
                "checkpoint_sha256": item["identity"]["checkpoint"]["sha256"],
                "overall": item["summary"]["overall"],
            }
            for path, item in zip(paths, sessions, strict=True)
        ],
    }
