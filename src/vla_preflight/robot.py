"""Generate reviewable LeRobot bring-up plans without connecting to a robot."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import shutil
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, field_validator

from . import __version__
from .contract import load_json
from .workflow_io import atomic_json, digest_json


class CameraConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str = "opencv"
    index_or_path: StrictStr | StrictInt
    width: StrictInt = Field(default=640, ge=1, le=8192)
    height: StrictInt = Field(default=480, ge=1, le=8192)
    fps: StrictInt = Field(default=30, ge=1, le=240)


class SafetyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    emergency_stop_tested: StrictBool = False
    workspace_cleared: StrictBool = False
    low_speed_first_run: StrictBool = True
    human_supervision: StrictBool = True
    notes: list[str] = Field(default_factory=list, max_length=30)


class RobotConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_.-]+$")
    robot_type: str = Field(min_length=1, max_length=100)
    robot_port: str = Field(min_length=1, max_length=260)
    robot_id: str = Field(min_length=1, max_length=100)
    teleop_type: str = Field(min_length=1, max_length=100)
    teleop_port: str = Field(min_length=1, max_length=260)
    teleop_id: str = Field(min_length=1, max_length=100)
    cameras: dict[str, CameraConfig] = Field(min_length=1, max_length=8)
    dataset_repo_id: str = Field(min_length=3, max_length=200)
    task: str = Field(min_length=1, max_length=500)
    num_episodes: StrictInt = Field(default=30, ge=2, le=10000)
    calibration_files: list[Path] = Field(default_factory=list, max_length=20)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)

    @field_validator("robot_type", "teleop_type", "robot_id", "teleop_id")
    @classmethod
    def safe_token(cls, value):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
            raise ValueError("must contain only letters, numbers, dot, underscore or dash")
        return value

    @field_validator("dataset_repo_id")
    @classmethod
    def repo_id(cls, value):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", value):
            raise ValueError("must look like owner/dataset")
        return value

    @field_validator("cameras")
    @classmethod
    def camera_names(cls, value):
        if any(not re.fullmatch(r"[A-Za-z0-9_.-]+", name) for name in value):
            raise ValueError("camera names must be simple identifiers")
        return value


def _digest(path: Path):
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def _common(config: RobotConfig):
    cameras = {name: camera.model_dump() for name, camera in config.cameras.items()}
    return [
        f"--robot.type={config.robot_type}",
        f"--robot.port={config.robot_port}",
        f"--robot.id={config.robot_id}",
        f"--robot.cameras={json.dumps(cameras, separators=(',', ':'))}",
        f"--teleop.type={config.teleop_type}",
        f"--teleop.port={config.teleop_port}",
        f"--teleop.id={config.teleop_id}",
    ]


def _ports():
    try:
        from serial.tools import list_ports
    except ImportError as exc:
        raise ValueError('Port probing needs: pip install "vla-preflight[robot]"') from exc
    return [item.device for item in list_ports.comports()]


def _camera(camera: CameraConfig):
    try:
        import cv2
    except ImportError as exc:
        raise ValueError('Camera probing needs: pip install "vla-preflight[robot]"') from exc
    capture = cv2.VideoCapture(camera.index_or_path)
    if not capture.isOpened():
        capture.release()
        raise ValueError("camera could not be opened")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, camera.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, camera.height)
    capture.set(cv2.CAP_PROP_FPS, camera.fps)
    frames = []
    started = time.monotonic()
    try:
        for _ in range(3):
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ValueError("camera did not return three readable frames")
            frames.append(frame)
    finally:
        capture.release()
    elapsed = max(time.monotonic() - started, 1e-9)
    height, width = frames[-1].shape[:2]
    return {
        "width": int(width),
        "height": int(height),
        "sampled_frames": len(frames),
        "sample_rate_fps": len(frames) / elapsed,
        "nonconstant": bool(float(frames[-1].std()) > 0),
    }


def check_robot(
    config_path: Path,
    *,
    port_provider=None,
    camera_probe=None,
    command_finder=None,
    probe_ports=True,
    probe_cameras=True,
):
    """Run non-actuating checks. Serial ports are enumerated but never opened."""
    config_path = config_path.resolve()
    config = RobotConfig.model_validate(load_json(config_path))
    port_provider = port_provider or _ports
    camera_probe = camera_probe or _camera
    command_finder = command_finder or shutil.which
    checks = []

    def add(code, status, message, **evidence):
        checks.append({"code": code, "status": status, "message": message, "evidence": evidence})

    calibration = []
    for declared in config.calibration_files:
        path = declared if declared.is_absolute() else config_path.parent / declared
        if path.is_file():
            digest = _digest(path)
            calibration.append({"declared_path": str(declared), "sha256": digest})
            add("CALIBRATION", "pass", "Calibration file is readable and hashed", sha256=digest)
        else:
            add("CALIBRATION", "fail", "Calibration file is missing", path=str(declared))
    if not config.calibration_files:
        add("CALIBRATION", "fail", "At least one calibration file must be declared")

    for name in ("lerobot-teleoperate", "lerobot-record"):
        found = command_finder(name)
        add(
            "LEROBOT_COMMAND",
            "pass" if found else "fail",
            f"{name} is {'available' if found else 'unavailable'}",
            command=name,
        )

    if probe_ports:
        try:
            available = set(port_provider())
            for role, port in (("robot", config.robot_port), ("teleop", config.teleop_port)):
                add(
                    "SERIAL_PORT",
                    "pass" if port in available else "fail",
                    f"Declared {role} port is {'present' if port in available else 'absent'}",
                    role=role,
                    port=port,
                )
        except (OSError, ValueError) as exc:
            add("SERIAL_PORT", "fail", str(exc))
    else:
        add("SERIAL_PORT", "skip", "Serial-port enumeration was explicitly skipped")

    if probe_cameras:
        for name, camera in config.cameras.items():
            try:
                observed = camera_probe(camera)
                shape_ok = observed["width"] == camera.width and observed["height"] == camera.height
                signal_ok = observed.get("nonconstant", True)
                add(
                    "CAMERA_FRAME",
                    "pass" if shape_ok and signal_ok else "fail",
                    f"Camera {name} returned frames",
                    camera=name,
                    expected=[camera.width, camera.height],
                    observed=observed,
                )
            except (OSError, ValueError) as exc:
                add("CAMERA_FRAME", "fail", f"Camera {name}: {exc}", camera=name)
    else:
        add("CAMERA_FRAME", "skip", "Camera capture was explicitly skipped")

    for code, passed, message in (
        ("EMERGENCY_STOP", config.safety.emergency_stop_tested, "Emergency stop tested"),
        ("WORKSPACE", config.safety.workspace_cleared, "Workspace cleared"),
        ("LOW_SPEED", config.safety.low_speed_first_run, "Low-speed first run selected"),
        ("SUPERVISION", config.safety.human_supervision, "Human supervision declared"),
    ):
        add(code, "pass" if passed else "fail", message)
    failures = [item for item in checks if item["status"] == "fail"]
    skipped = [item for item in checks if item["status"] == "skip"]
    return {
        "schema": "vla-preflight.robot-check/1",
        "tool_version": __version__,
        "config_digest": digest_json(config.model_dump(mode="json")),
        "status": "passed"
        if not failures and not skipped
        else "incomplete"
        if skipped
        else "failed",
        "ready": not failures and not skipped,
        "checks": checks,
        "calibration": calibration,
        "coverage": {
            "ports": "enumerated-not-opened" if probe_ports else "skipped",
            "cameras": "three-frames" if probe_cameras else "skipped",
            "motors": "not-contacted",
            "emergency_stop": "operator-attestation-only",
        },
    }


def write_robot_check(config_path: Path, destination: Path, **options):
    report = check_robot(config_path, **options)
    atomic_json(destination, report)
    return report


def create_robot_plan(config_path: Path, destination: Path, *, preflight: Path | None = None):
    config_path = config_path.resolve()
    config = RobotConfig.model_validate(load_json(config_path))
    destination = destination.resolve()
    if destination.exists():
        raise ValueError(f"Output already exists; choose a new directory: {destination}")
    destination.mkdir(parents=True)
    calibration = []
    blockers = []
    if preflight is None:
        blockers.append("A passing robot-check report is required")
    else:
        report = load_json(preflight)
        if report.get("schema") != "vla-preflight.robot-check/1":
            blockers.append("Robot-check report schema is unsupported")
        elif report.get("config_digest") != digest_json(config.model_dump(mode="json")):
            blockers.append("Robot-check report belongs to a different configuration")
        elif not report.get("ready"):
            blockers.append("Robot-check report is not ready")
    for declared in config.calibration_files:
        path = declared if declared.is_absolute() else config_path.parent / declared
        if not path.is_file():
            blockers.append(f"Calibration file is missing: {declared}")
        else:
            calibration.append({"declared_path": str(declared), "sha256": _digest(path)})
    safety = config.safety
    if not safety.emergency_stop_tested:
        blockers.append("Emergency stop has not been marked as tested")
    if not safety.workspace_cleared:
        blockers.append("Robot workspace has not been marked as cleared")
    if not safety.low_speed_first_run:
        blockers.append("First run is not configured as low speed")
    if not safety.human_supervision:
        blockers.append("Human supervision is required for initial bring-up")
    common = _common(config)
    commands = {
        "teleoperate": ["lerobot-teleoperate", *common, "--display_data=true"],
        "record": [
            "lerobot-record",
            *common,
            f"--dataset.repo_id={config.dataset_repo_id}",
            f"--dataset.num_episodes={config.num_episodes}",
            f"--dataset.single_task={config.task}",
            "--dataset.streaming_encoding=true",
            "--display_data=true",
        ],
    }
    result = {
        "schema": "vla-preflight.robot-plan/1",
        "tool_version": __version__,
        "status": "ready-for-supervised-bringup" if not blockers else "blocked",
        "safe_to_start": not blockers,
        "blockers": blockers,
        "config": config.model_dump(mode="json"),
        "calibration": calibration,
        "preflight_report": str(preflight.resolve()) if preflight else None,
        "commands": commands,
        "limits": [
            "Generated commands are never executed by this tool.",
            "A passing robot-check covers ports and camera frames without contacting motors.",
            "Calibration correctness and emergency-stop behavior still need physical verification.",
            "Inspect a short teleoperation recording before training or policy rollout.",
        ],
    }
    atomic_json(destination / "plan.json", result)
    lines = [
        f"# Robot bring-up plan: {config.name}",
        "",
        f"Status: **{result['status']}**",
        "",
        "## Blocking gates",
        "",
        *(f"- {item}" for item in blockers),
    ]
    if not blockers:
        lines.append("- None declared. Physical checks are still required.")
    lines.extend(["", "## Reviewable commands", ""])
    for name, command in commands.items():
        lines.extend([f"### {name}", "", "```shell", shlex.join(command), "```", ""])
    lines.extend(
        [
            "## Required order",
            "",
            "1. Verify the emergency stop and clear the workspace.",
            "2. Run supervised low-speed teleoperation and inspect every camera.",
            "3. Record a short disposable dataset and run `vla-preflight audit`.",
            "4. Train and evaluate offline before any policy rollout.",
            "5. Log every physical rollout and summarize it with `vla-preflight rollout-eval`.",
        ]
    )
    (destination / "RUNBOOK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result
