"""A real, deliberately small VLA reference trainer and reproducible run artifacts.

Randomly initialized CNN + byte-text encoder + state encoder, not a foundation model.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictInt

from . import __version__
from .dataset import rows
from .media import read_image, visual_keys
from .workflow_io import atomic_json, digest_json, load_prepared, new_artifact_dir, tasks


class TrainOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    steps: StrictInt = Field(default=200, ge=1, le=100000)
    batch_size: StrictInt = Field(default=16, ge=1, le=512)
    chunk_size: StrictInt = Field(default=4, ge=1, le=64)
    learning_rate: float = Field(default=0.001, gt=0, le=0.1, allow_inf_nan=False)
    seed: StrictInt = Field(default=7, ge=0, le=2**31 - 1)
    log_every: StrictInt = Field(default=20, ge=1, le=10000)
    max_frames: StrictInt = Field(default=50000, ge=1, le=500000)
    device: str = "cpu"
    camera: str | None = None


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ValueError(
            'Reference training requires the optional train dependencies: pip install -e ".[train]"'
        ) from exc
    return torch


def build_model(state_dim, action_dim, chunk_size):
    torch = _torch()
    nn = torch.nn

    class TinyVLA(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision = nn.Sequential(
                nn.Conv2d(3, 12, 3, padding=1),
                nn.ReLU(),
                nn.AvgPool2d(2),
                nn.Conv2d(12, 24, 3, padding=1),
                nn.ReLU(),
                nn.AvgPool2d(2),
                nn.Flatten(),
                nn.Linear(24 * 8 * 8, 64),
                nn.ReLU(),
            )
            self.language = nn.Embedding(257, 24, padding_idx=0)
            self.state = nn.Sequential(nn.Linear(state_dim, 24), nn.ReLU())
            self.head = nn.Sequential(
                nn.Linear(112, 96), nn.ReLU(), nn.Linear(96, action_dim * chunk_size)
            )

        def forward(self, images, tokens, state):
            mask = (tokens != 0).unsqueeze(-1)
            language = (self.language(tokens) * mask).sum(1) / mask.sum(1).clamp(min=1)
            combined = torch.cat((self.vision(images), language, self.state(state)), dim=-1)
            return self.head(combined).reshape(-1, chunk_size, action_dim)

    return TinyVLA()


def load_training_data(ds, normal, camera, max_frames):
    torch = _torch()
    if "observation.state" not in ds.features:
        raise ValueError("Reference trainer requires observation.state")
    keys = visual_keys(ds)
    if not keys:
        raise ValueError(
            "Reference VLA requires visual input; numeric-only data are not a VLA demo"
        )
    camera = camera or keys[0]
    if camera not in keys:
        raise ValueError("Selected camera is absent")
    if sum(m["length"] for m in ds.episodes.values()) > max_frames:
        raise ValueError(
            "Dataset exceeds max_frames; export a smaller subset or increase the explicit limit"
        )
    texts = tasks(ds)
    images, tokens, states, actions, episodes, frames = [], [], [], [], [], []
    columns = [
        "episode_index",
        "frame_index",
        "timestamp",
        "task_index",
        "action",
        "observation.state",
    ]
    if ds.features[camera]["dtype"] == "image":
        columns.append(camera)
    for path in ds.data_files(set(ds.episodes)):
        for row in rows(path, columns):
            ep = row["episode_index"]
            instruction = texts.get(row.get("task_index"))
            if not instruction or not instruction.strip():
                raise ValueError("Every frame needs a nonempty task instruction in task metadata")
            images.append(
                np.asarray(read_image(ds, row, camera), dtype=np.uint8).transpose(2, 0, 1)
            )
            token = np.zeros(64, dtype=np.int64)
            encoded = (
                np.frombuffer(instruction.encode("utf-8")[:64], dtype=np.uint8).astype(np.int64) + 1
            )
            token[: len(encoded)] = encoded
            tokens.append(token)
            states.append(row["observation.state"])
            actions.append(row["action"])
            episodes.append(ep)
            frames.append(row["frame_index"])
    data = {
        "images": torch.from_numpy(np.stack(images)),
        "tokens": torch.from_numpy(np.stack(tokens)),
        "episodes": np.asarray(episodes),
        "frames": np.asarray(frames),
    }
    for name, values, key in (
        ("states", states, "observation.state"),
        ("actions", actions, "action"),
    ):
        fit = normal["features"][key]
        mean = np.asarray(fit["mean"], dtype=np.float32)
        scale = np.maximum(np.asarray(fit["std"], dtype=np.float32), normal["epsilon"])
        array = np.asarray(values, dtype=np.float32)
        data[name] = torch.from_numpy((array - mean) / scale)
        data[name + "_raw"] = torch.from_numpy(array)
        data[name + "_mean"] = torch.from_numpy(mean)
        data[name + "_scale"] = torch.from_numpy(scale)
    data["lookup"] = {(ep, fr): i for i, (ep, fr) in enumerate(zip(episodes, frames, strict=True))}
    data["camera"] = camera
    return data


def batch(data, indices, ds, chunk_size, device):
    torch = _torch()
    targets, padding, traces = [], [], []
    for index in indices:
        ep, frame = int(data["episodes"][index]), int(data["frames"][index])
        length = ds.episodes[ep]["length"]
        target_frames = [min(frame + offset, length - 1) for offset in range(chunk_size)]
        mask = [frame + offset >= length for offset in range(chunk_size)]
        targets.append([data["lookup"][ep, fr] for fr in target_frames])
        padding.append(mask)
        traces.append(
            {
                "episode_index": ep,
                "anchor_frame": frame,
                "target_episode_indices": [ep] * chunk_size,
                "target_frame_indices": target_frames,
                "padding_mask": mask,
            }
        )
    return (
        data["images"][indices].float().to(device) / 255,
        data["tokens"][indices].to(device),
        data["states"][indices].to(device),
        data["actions"][torch.tensor(targets, dtype=torch.long)].to(device),
        torch.tensor(padding, device=device),
        traces,
    )


def evaluate(model, data, indices, device):
    torch = _torch()
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(indices), 64):
            idx = indices[start : start + 64]
            prediction = model(
                data["images"][idx].float().to(device) / 255,
                data["tokens"][idx].to(device),
                data["states"][idx].to(device),
            )[:, 0]
            predictions.append(prediction.cpu() * data["actions_scale"] + data["actions_mean"])
    pred = torch.cat(predictions)
    target = data["actions_raw"][indices]
    error = pred - target
    return {
        "mae": float(error.abs().mean()),
        "rmse": float(error.square().mean().sqrt()),
        "per_dimension_mae": error.abs().mean(0).tolist(),
        "mean_action_baseline_mae": float((target - data["actions_mean"]).abs().mean()),
        "zero_action_baseline_mae": float(target.abs().mean()),
        "frames": len(indices),
    }, pred


def train(
    prepared: Path,
    destination: Path,
    *,
    options: TrainOptions | None = None,
    resume: Path | None = None,
    progress=None,
    cancelled=None,
):
    options = options or TrainOptions()
    torch = _torch()
    if options.device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if options.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in this PyTorch installation")
    ds, manifest, split, normal = load_prepared(prepared)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(options.seed)
    rng = np.random.default_rng(options.seed)
    data = load_training_data(ds, normal, options.camera, options.max_frames)
    train_ids = np.flatnonzero(np.isin(data["episodes"], split["train"]))
    val_ids = np.flatnonzero(np.isin(data["episodes"], split["validation"]))
    if not len(train_ids) or not len(val_ids):
        raise ValueError("Training and validation frames must both be present")
    model_config = {
        "state_dim": data["states"].shape[1],
        "action_dim": data["actions"].shape[1],
        "chunk_size": options.chunk_size,
    }
    model = build_model(**model_config).to(options.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=options.learning_rate)
    bundle_digest = digest_json(
        {"source": manifest["source"]["digest"], "artifacts": manifest["artifact_hashes"]}
    )
    start_step = 0
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=True)
        if (
            checkpoint["bundle_digest"] != bundle_digest
            or checkpoint["model_config"] != model_config
        ):
            raise ValueError("Checkpoint was trained on a different prepared bundle or model shape")
        if checkpoint["camera"] != data["camera"]:
            raise ValueError("Checkpoint camera differs")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        rng.bit_generator.state = checkpoint["numpy_rng"]
        torch.set_rng_state(checkpoint["torch_rng"])
        start_step = checkpoint["step"]
        if options.steps <= start_step:
            raise ValueError("steps is the total target; it must exceed the checkpoint step")
        for group in optimizer.param_groups:
            group["lr"] = options.learning_rate
    destination = new_artifact_dir(destination, ds.root)
    info = {
        "schema": "vla-preflight.run/1",
        "tool_version": __version__,
        "backend": "tiny-vla",
        "status": "running",
        "options": options.model_dump(),
        "bundle_digest": bundle_digest,
        "prepared": str(prepared.resolve()),
        "parameters": sum(p.numel() for p in model.parameters()),
        "model": model_config,
        "camera": data["camera"],
        "train_frames": len(train_ids),
        "validation_frames": len(val_ids),
        "start_step": start_step,
        "limits": "Small randomly initialized VLA; offline action error is not task success",
    }
    atomic_json(destination / "run.json", info)
    started = time.monotonic()
    step = start_step

    def save_checkpoint(name="checkpoint.pt"):
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "model_config": model_config,
                "bundle_digest": bundle_digest,
                "camera": data["camera"],
                "numpy_rng": rng.bit_generator.state,
                "torch_rng": torch.get_rng_state(),
            },
            destination / (name + ".tmp"),
        )
        os.replace(destination / (name + ".tmp"), destination / name)

    try:
        initial, _ = evaluate(model, data, val_ids, options.device)
        best = {"step": start_step, **initial}
        save_checkpoint("best.pt")
        with (
            (destination / "metrics.jsonl").open("w", encoding="utf-8") as log,
            (destination / "sampler-trace.jsonl").open("w", encoding="utf-8") as trace,
        ):
            for step in range(start_step + 1, options.steps + 1):
                if cancelled and cancelled():
                    step -= 1
                    info["status"] = "cancelled"
                    break
                model.train()
                indices = rng.choice(train_ids, size=options.batch_size, replace=True)
                images, tokens, states, target, mask, traces = batch(
                    data, indices, ds, options.chunk_size, options.device
                )
                optimizer.zero_grad(set_to_none=True)
                prediction = model(images, tokens, states)
                valid = (~mask).unsqueeze(-1).expand_as(target)
                loss = ((prediction - target).square() * valid).sum() / valid.sum()
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(norm):
                    raise ValueError("Nonfinite gradient norm")
                optimizer.step()
                if step == start_step + 1:
                    for item in traces:
                        trace.write(json.dumps(item) + "\n")
                if step % options.log_every == 0 or step == options.steps or step == start_step + 1:
                    metrics, _ = evaluate(model, data, val_ids, options.device)
                    if metrics["mae"] < best["mae"]:
                        best = {"step": step, **metrics}
                        save_checkpoint("best.pt")
                    item = {
                        "step": step,
                        "train_loss": float(loss.detach()),
                        "validation_mae": metrics["mae"],
                        "gradient_norm": float(norm),
                        "seconds": time.monotonic() - started,
                    }
                    log.write(json.dumps(item, allow_nan=False) + "\n")
                    log.flush()
                    save_checkpoint()
                    if progress:
                        progress(item)
            save_checkpoint()
        final, predictions = evaluate(model, data, val_ids, options.device)
        sample = np.linspace(0, len(val_ids) - 1, min(200, len(val_ids))).astype(int)
        atomic_json(
            destination / "predictions.json",
            {
                "scope": "held-out first action in each chunk, decimated preview",
                "episode": data["episodes"][val_ids[sample]].tolist(),
                "frame": data["frames"][val_ids[sample]].tolist(),
                "target": data["actions_raw"][val_ids[sample]].tolist(),
                "prediction": predictions[sample].tolist(),
            },
        )
        info.update(
            {
                "status": "cancelled" if info["status"] == "cancelled" else "completed",
                "completed_steps": step,
                "initial_validation": initial,
                "validation": final,
                "best_validation": best,
                "seconds": time.monotonic() - started,
            }
        )
        info["diagnostics"] = []
        if final["mae"] > initial["mae"]:
            info["diagnostics"].append(
                "Validation MAE worsened versus initialization; "
                "inspect overfitting and data coverage."
            )
        if final["mae"] >= min(
            final["mean_action_baseline_mae"], final["zero_action_baseline_mae"]
        ):
            info["diagnostics"].append(
                "Final model did not beat both constant-action baselines on held-out episodes."
            )
        # Roundtrip check uses actual train-only normalizer, not just declarations.
        x = data["actions_raw"].numpy().astype(np.float64)
        mean, scale = data["actions_mean"].numpy(), data["actions_scale"].numpy()
        info["normalization_roundtrip_max_error"] = float(
            np.max(np.abs(((x - mean) / scale) * scale + mean - x))
        )
        atomic_json(destination / "run.json", info)
        return info
    except BaseException as exc:
        info.update(
            {
                "status": "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed",
                "completed_steps": step,
                "error": str(exc),
            }
        )
        atomic_json(destination / "run.json", info)
        raise


def compare_runs(paths):
    from .contract import load_json

    runs = [load_json(path / "run.json") for path in paths]
    if len(runs) < 2:
        raise ValueError("Provide at least two runs")
    comparable = len({r.get("bundle_digest") for r in runs}) == 1
    return {
        "same_prepared_bundle": comparable,
        "note": "Offline MAE is comparable only on the same held-out split and preprocessing",
        "runs": [
            {
                "name": p.name,
                "status": r["status"],
                "backend": r["backend"],
                "steps": r.get("completed_steps"),
                "parameters": r.get("parameters"),
                "seconds": r.get("seconds"),
                "validation": r.get("validation"),
            }
            for p, r in zip(paths, runs, strict=True)
        ],
    }
