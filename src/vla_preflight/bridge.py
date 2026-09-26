"""Explicit SmolVLA launch plans. Planning never downloads a model or starts training."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt

from .contract import load_json
from .dataset import Dataset
from .export import export_episodes
from .workflow_io import atomic_json, fingerprint, load_prepared, new_artifact_dir


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


def create_smol_plan(prepared: Path, destination: Path, *, options: SmolOptions | None = None):
    options = options or SmolOptions()
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
        "limits": [
            "SmolVLA execution requires a separate compatible LeRobot environment.",
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
        plan.update(
            status="completed" if code == 0 and checkpoints else "failed",
            returncode=code,
            seconds=time.monotonic() - started,
            checkpoint_files=[str(p.relative_to(plan_dir)) for p in checkpoints],
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
    return plan
