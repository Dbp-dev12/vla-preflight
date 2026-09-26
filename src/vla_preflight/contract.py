"""An explicit audit contract, deliberately separate from a trainer configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator


class Semantics(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action_names: list[str] | None = None
    action_units: list[str] | None = None
    action_mode: Literal["absolute", "delta", "velocity"] | None = None
    coordinate_frame: str | None = None
    gripper_convention: str | None = None


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    action_dim: StrictInt | None = Field(default=None, gt=0)
    state_dim: StrictInt | None = Field(default=None, gt=0)
    fps: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    camera_keys: list[str] = Field(default_factory=list)
    normalization: dict[
        Literal["action", "observation.state"],
        Literal["IDENTITY", "MEAN_STD", "MIN_MAX", "QUANTILES"],
    ] = Field(default_factory=dict)
    train_episodes: list[StrictInt] | None = None
    validation_episodes: list[StrictInt] | None = None
    training: Semantics | None = None
    deployment: Semantics | None = None

    @model_validator(mode="after")
    def validate_lists(self):
        for name in ("train_episodes", "validation_episodes"):
            values = getattr(self, name)
            if values is not None and (not values or any(x < 0 for x in values)):
                raise ValueError(f"{name} must be nonempty and contain nonnegative integers")
            if values is not None and len(set(values)) != len(values):
                raise ValueError(f"{name} contains duplicates")
        for name in ("training", "deployment"):
            sem = getattr(self, name)
            if sem and self.action_dim:
                for key in ("action_names", "action_units"):
                    value = getattr(sem, key)
                    if value is not None and len(value) != self.action_dim:
                        raise ValueError(f"{name}.{key} must match action_dim")
        return self


def load_json(path: Path):
    with path.open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def load_contract(path: Path) -> Contract:
    return Contract.model_validate(load_json(path))


def from_lerobot(path: Path) -> Contract:
    """Read resolved, serialized LeRobot train config, never instantiate the trainer.

    Camera remapping and custom processors are not inferred. A separate explicit
    contract is required for those pipelines.
    """
    obj = load_json(path)
    if not isinstance(obj, dict) or not isinstance(obj.get("policy"), dict):
        raise ValueError("Expected a resolved LeRobot JSON train config with a policy object")
    policy = obj["policy"]
    features = {}
    for name in ("input_features", "output_features"):
        value = policy.get(name) or {}
        if not isinstance(value, dict):
            raise ValueError(f"policy.{name} must be an object")
        features.update(value)
    if not features:
        raise ValueError("Unresolved config: policy input_features/output_features are empty")
    params: dict = {"camera_keys": [], "normalization": {}}
    mapping = policy.get("normalization_mapping") or {}
    if not isinstance(mapping, dict):
        raise ValueError("policy.normalization_mapping must be an object")
    for name, feat in features.items():
        if not isinstance(feat, dict):
            raise ValueError(f"Invalid feature definition: {name}")
        kind = str(feat.get("type", "")).upper()
        shape = feat.get("shape", [])
        if kind == "VISUAL":
            params["camera_keys"].append(name)
        if name in ("action", "observation.state"):
            if not isinstance(shape, list) or len(shape) != 1:
                raise ValueError(f"Expected vector shape for {name}")
            params["action_dim" if name == "action" else "state_dim"] = shape[0]
            if kind in mapping:
                params["normalization"][name] = str(mapping[kind]).upper()
    if "action_dim" not in params:
        raise ValueError("Resolved config must include output_features.action")
    dataset = obj.get("dataset") or {}
    if not isinstance(dataset, dict):
        raise ValueError("dataset must be an object")
    if dataset.get("episodes") is not None:
        params["train_episodes"] = dataset["episodes"]
    return Contract.model_validate(params)
