"""Explicit SmolVLA launch plans. Planning never downloads a model or starts training."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt

from . import __version__
from .contract import load_json
from .dataset import Dataset
from .export import export_episodes
from .workflow_io import (
    atomic_json,
    digest_json,
    file_hash,
    fingerprint,
    load_prepared,
    new_artifact_dir,
)

SUPPORTED_LEROBOT = "0.6"


class SmolOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    python: str = sys.executable
    pretrained: str = "lerobot/smolvla_base"
    steps: StrictInt = Field(default=1000, ge=1, le=1000000)
    batch_size: StrictInt = Field(default=1, ge=1, le=1024)
    gradient_accumulation: StrictInt = Field(default=8, ge=1, le=1024)
    gradient_checkpointing: StrictBool = True
    seed: StrictInt = Field(default=7, ge=0, le=2**31 - 1)
    device: str = "cuda"


def smol_command(plan_dir: Path, options: SmolOptions):
    if options.device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    return [
        options.python,
        "-m",
        "lerobot.scripts.lerobot_train",
        f"--policy.path={options.pretrained}",
        "--dataset.repo_id=local/preflight-train",
        f"--dataset.root={plan_dir.resolve() / 'train-dataset'}",
        f"--output_dir={plan_dir.resolve() / 'artifacts'}",
        f"--steps={options.steps}",
        f"--batch_size={options.batch_size}",
        f"--accelerator.gradient_accumulation.steps={options.gradient_accumulation}",
        f"--seed={options.seed}",
        f"--policy.device={options.device}",
        "--policy.freeze_vision_encoder=true",
        "--policy.train_expert_only=true",
        f"--policy.gradient_checkpointing={str(options.gradient_checkpointing).lower()}",
        "--policy.push_to_hub=false",
        "--wandb.enable=false",
        "--num_workers=0",
        "--save_freq=500",
        "--log_freq=10",
    ]


def inspect_lerobot(python: str, *, device="cuda", runner=None):
    """Inspect a separate LeRobot environment without importing it in this process."""
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    script = r"""
import importlib.metadata, importlib.util, json, platform, sys
result = {
    "python": platform.python_version(),
    "executable": sys.executable,
    "lerobot_version": importlib.metadata.version("lerobot"),
    "train_module": importlib.util.find_spec("lerobot.scripts.lerobot_train") is not None,
    "smolvla_module": importlib.util.find_spec("lerobot.policies.smolvla") is not None,
}
try:
    import torch
    result.update(
        torch_version=torch.__version__,
        cuda_available=bool(torch.cuda.is_available()),
        cuda_devices=[
            {
                "name": torch.cuda.get_device_properties(i).name,
                "total_memory_gib": round(
                    torch.cuda.get_device_properties(i).total_memory / 1024**3, 2
                ),
            }
            for i in range(torch.cuda.device_count())
        ],
    )
except Exception as exc:
    result.update(cuda_available=False, cuda_devices=[], torch_error=type(exc).__name__)
print(json.dumps(result))
"""
    runner = runner or subprocess.run
    try:
        completed = runner(
            [python, "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f"LeRobot environment probe failed: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip().splitlines()[-1:]
        raise ValueError(
            "LeRobot environment probe failed: " + (detail[0] if detail else "unknown")
        )
    try:
        observed = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise ValueError("LeRobot environment probe returned invalid JSON") from exc
    checks = {
        "python_3_12_plus": tuple(map(int, observed["python"].split(".")[:2])) >= (3, 12),
        "supported_lerobot": observed["lerobot_version"].startswith(SUPPORTED_LEROBOT + "."),
        "train_module": bool(observed["train_module"]),
        "smolvla_module": bool(observed["smolvla_module"]),
        "requested_device": device == "cpu" or bool(observed["cuda_available"]),
    }
    return {
        "schema": "vla-preflight.lerobot-environment/1",
        "tool_version": __version__,
        "status": "compatible" if all(checks.values()) else "incompatible",
        "compatible": all(checks.values()),
        "requested_device": device,
        "checks": checks,
        "observed": observed,
        "supported_lerobot_series": SUPPORTED_LEROBOT + ".x",
    }


def write_lerobot_environment(python: str, destination: Path, *, device="cuda"):
    report = inspect_lerobot(python, device=device)
    atomic_json(destination, report)
    return report


def _environment(path: Path, options: SmolOptions):
    report = load_json(path)
    if report.get("schema") != "vla-preflight.lerobot-environment/1":
        raise ValueError("Unsupported LeRobot environment report")
    if not report.get("compatible"):
        raise ValueError("LeRobot environment report is incompatible")
    if report.get("requested_device") != options.device:
        raise ValueError("LeRobot environment report checked a different device")
    return report


def create_smol_plan(
    prepared: Path,
    destination: Path,
    *,
    options: SmolOptions | None = None,
    environment_report: Path | None = None,
):
    options = options or SmolOptions()
    if environment_report is None:
        raise ValueError("A compatible lerobot-check report is required")
    environment = _environment(environment_report, options)
    ds, manifest, split, _ = load_prepared(prepared)
    if not any(v.get("dtype") in ("image", "video") for v in ds.features.values()):
        raise ValueError("SmolVLA needs visual observations")
    smol_command(destination, options)  # validate before export
    destination = new_artifact_dir(destination, ds.root)
    export_episodes(ds.root, destination / "train-dataset", exclude=set(split["validation"]))
    source = fingerprint(Dataset(destination / "train-dataset"), videos=True)
    plan = {
        "schema": "vla-preflight.smol-plan/1",
        "status": "planned",
        "options": options.model_dump(),
        "command_preview": smol_command(destination, options),
        "training_data_digest": source["digest"],
        "original_source": manifest["source"]["digest"],
        "held_out_original_episode_ids": split["validation"],
        "normalization": "Recomputed on physically exported training episodes only",
        "effective_batch_size": options.batch_size * options.gradient_accumulation,
        "environment": environment,
        "environment_digest": digest_json(environment),
        "limits": [
            "Execution is delegated to the separately checked LeRobot environment.",
            "Model memory use has not been measured; validate it on the target system.",
            "This adapter launches fine-tuning; external model evaluation is not implemented.",
            "Only the built-in tiny-vla backend has end-to-end local validation in v0.6.",
        ],
    }
    atomic_json(destination / "plan.json", plan)
    return plan


def launch_smol(plan_dir: Path):
    plan = load_json(plan_dir / "plan.json")
    if plan.get("schema") != "vla-preflight.smol-plan/1" or plan.get("status") != "planned":
        raise ValueError("Only a new, unexecuted SmolVLA plan may be launched")
    options = SmolOptions.model_validate(plan["options"])
    source = fingerprint(Dataset(plan_dir / "train-dataset"), videos=True)
    if source["digest"] != plan["training_data_digest"]:
        raise ValueError("Training dataset changed since planning")
    if (plan_dir / "artifacts").exists():
        raise ValueError("Training artifacts directory already exists")
    command = smol_command(plan_dir, options)
    plan.update(status="running", command_preview=command)
    atomic_json(plan_dir / "plan.json", plan)
    started = time.monotonic()
    process = None
    try:
        with (plan_dir / "external.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=plan_dir, stdout=log, stderr=subprocess.STDOUT)
            code = process.wait()
        checkpoints = list((plan_dir / "artifacts").rglob("*.safetensors"))
        checkpoint_files = [
            {
                "path": str(path.relative_to(plan_dir)),
                "size": path.stat().st_size,
                "sha256": file_hash(path),
            }
            for path in checkpoints
        ]
        log_text = (plan_dir / "external.log").read_text(encoding="utf-8", errors="replace")
        memory = []
        for value, unit in re.findall(
            r"(?i)peak[^\n]{0,80}?memory[^\n]{0,40}?([0-9]+(?:\.[0-9]+)?)\s*(GiB|GB|MiB|MB)",
            log_text,
        ):
            gib = float(value) / 1024 if unit.lower() in ("mib", "mb") else float(value)
            memory.append(gib)
        plan.update(
            status="completed" if code == 0 and checkpoints else "failed",
            returncode=code,
            seconds=time.monotonic() - started,
            checkpoint_files=checkpoint_files,
            measured_peak_memory_gib=max(memory) if memory else None,
        )
        if code == 0 and not checkpoints:
            plan["error"] = "Process exited successfully but no safetensors checkpoint was found"
    except BaseException as exc:
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        plan.update(
            status="cancelled" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc)
        )
        atomic_json(plan_dir / "plan.json", plan)
        raise
    atomic_json(plan_dir / "plan.json", plan)
    atomic_json(
        plan_dir / "external-run.json",
        {
            "schema": "vla-preflight.external-run/1",
            "status": plan["status"],
            "returncode": plan.get("returncode"),
            "seconds": plan.get("seconds"),
            "training_data_digest": plan["training_data_digest"],
            "environment_digest": plan["environment_digest"],
            "checkpoint_files": plan.get("checkpoint_files", []),
            "measured_peak_memory_gib": plan.get("measured_peak_memory_gib"),
            "log_file": "external.log",
        },
    )
    return plan
