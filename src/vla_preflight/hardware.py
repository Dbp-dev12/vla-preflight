"""Privacy-conscious compute diagnostics for local training decisions."""

from __future__ import annotations

import importlib.metadata
import platform
import shutil
from pathlib import Path

from . import __version__
from .workflow_io import atomic_json


def _package(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _cuda():
    result = {"available": False, "devices": []}
    try:
        import torch
    except ImportError:
        result["reason"] = "PyTorch is not installed"
        return result
    result["torch_version"] = torch.__version__
    result["torch_cuda_version"] = torch.version.cuda
    result["available"] = bool(torch.cuda.is_available())
    if not result["available"]:
        result["reason"] = "This PyTorch build cannot access CUDA"
        return result
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        result["devices"].append(
            {
                "index": index,
                "name": props.name,
                "total_memory_gib": round(props.total_memory / 1024**3, 2),
                "capability": list(torch.cuda.get_device_capability(index)),
                "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            }
        )
    return result


def _recommend(cuda):
    if not cuda["available"]:
        return {
            "tier": "cpu",
            "guidance": [
                "Run audit, preparation, export and the Tiny VLA smoke path on CPU.",
                "Use a CUDA machine for pretrained VLA fine-tuning.",
            ],
        }
    memory = min(device["total_memory_gib"] for device in cuda["devices"])
    if memory < 8:
        tier = "constrained-cuda"
        guidance = [
            "Start with batch size 1, frozen vision/VLM components and gradient accumulation.",
            "Treat pretrained-model fit as unverified until a real dry run records peak memory.",
        ]
    elif memory < 16:
        tier = "single-gpu"
        guidance = [
            "Start pretrained fine-tuning with batch size 1 and gradient checkpointing.",
            "Increase the micro-batch only after recording measured peak memory.",
        ]
    else:
        tier = "roomy-single-gpu"
        guidance = [
            "Use a measured pilot run before selecting the final batch and image resolution.",
            "Keep gradient checkpointing available for larger policies or more cameras.",
        ]
    return {"tier": tier, "guidance": guidance}


def doctor():
    """Return a shareable report without hostnames, usernames or filesystem paths."""
    cuda = _cuda()
    return {
        "schema": "vla-preflight.doctor/1",
        "tool_version": __version__,
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "os": platform.system(),
        "architecture": platform.machine(),
        "packages": {
            name: _package(name)
            for name in ("vla-preflight", "torch", "lerobot", "numpy", "pyarrow", "av")
        },
        "commands": {
            name: bool(shutil.which(name))
            for name in ("ffmpeg", "ffprobe", "lerobot-train", "lerobot-record", "lerobot-rollout")
        },
        "cuda": cuda,
        "recommendation": _recommend(cuda),
        "privacy": "Machine identity, environment variables and absolute paths are omitted.",
    }


def write_doctor(destination: Path):
    report = doctor()
    atomic_json(destination, report)
    return report
