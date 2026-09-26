import json
from types import SimpleNamespace

import pytest

from vla_preflight.bridge import inspect_lerobot
from vla_preflight.cli import main
from vla_preflight.hardware import doctor
from vla_preflight.robot import check_robot, create_robot_plan, write_robot_check
from vla_preflight.rollouts import (
    compare_rollout_sessions,
    create_rollout_session,
    evaluate_rollouts,
    write_rollout_evaluation,
)


def robot_config(tmp_path, **safety):
    calibration = tmp_path / "calibration.json"
    calibration.write_text('{"zero": [0, 0]}', encoding="utf-8")
    config = {
        "name": "test-arm",
        "robot_type": "so101_follower",
        "robot_port": "COM5",
        "robot_id": "follower",
        "teleop_type": "so101_leader",
        "teleop_port": "COM6",
        "teleop_id": "leader",
        "cameras": {"front": {"index_or_path": 0, "width": 640, "height": 480, "fps": 30}},
        "dataset_repo_id": "owner/dataset",
        "task": "move the cube",
        "num_episodes": 10,
        "calibration_files": ["calibration.json"],
        "safety": {
            "emergency_stop_tested": True,
            "workspace_cleared": True,
            "low_speed_first_run": True,
            "human_supervision": True,
            **safety,
        },
    }
    path = tmp_path / "robot.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_doctor_omits_machine_identity():
    result = doctor()
    text = json.dumps(result).lower()
    assert result["schema"] == "vla-preflight.doctor/1"
    assert "hostname" not in text and "username" not in text
    assert result["recommendation"]["tier"]


def test_lerobot_environment_contract():
    observed = {
        "python": "3.12.9",
        "executable": "/env/python",
        "lerobot_version": "0.6.2",
        "train_module": True,
        "smolvla_module": True,
        "torch_version": "2.9.0",
        "cuda_available": True,
        "cuda_devices": [{"name": "test", "total_memory_gib": 12}],
    }

    def runner(*args, **kwargs):
        assert kwargs["timeout"] == 30 and args[0][1] == "-c"
        return SimpleNamespace(returncode=0, stdout=json.dumps(observed), stderr="")

    result = inspect_lerobot("/env/python", runner=runner)
    assert result["compatible"] and all(result["checks"].values())
    observed["lerobot_version"] = "0.5.1"
    result = inspect_lerobot("/env/python", runner=runner)
    assert not result["compatible"] and not result["checks"]["supported_lerobot"]


def test_robot_plan_gates_and_hashes_calibration(tmp_path):
    config = robot_config(tmp_path)
    report = check_robot(
        config,
        port_provider=lambda: ["COM5", "COM6"],
        camera_probe=lambda _: {
            "width": 640,
            "height": 480,
            "sampled_frames": 3,
            "sample_rate_fps": 30,
            "nonconstant": True,
        },
        command_finder=lambda _: "available",
    )
    preflight = tmp_path / "preflight.json"
    preflight.write_text(json.dumps(report), encoding="utf-8")
    assert report["ready"] and report["coverage"]["motors"] == "not-contacted"
    result = create_robot_plan(config, tmp_path / "plan", preflight=preflight)
    assert result["safe_to_start"]
    assert len(result["calibration"][0]["sha256"]) == 64
    assert result["commands"]["record"][0] == "lerobot-record"
    assert "--dataset.num_episodes=10" in result["commands"]["record"]
    assert (tmp_path / "plan/RUNBOOK.md").is_file()
    with pytest.raises(ValueError, match="already exists"):
        create_robot_plan(config, tmp_path / "plan", preflight=preflight)


def test_robot_plan_blocks_unverified_safety(tmp_path):
    config = robot_config(tmp_path, emergency_stop_tested=False)
    report = check_robot(
        config,
        port_provider=lambda: ["COM5", "COM6"],
        camera_probe=lambda _: {
            "width": 640,
            "height": 480,
            "sampled_frames": 3,
            "sample_rate_fps": 30,
            "nonconstant": True,
        },
        command_finder=lambda _: "available",
    )
    preflight = tmp_path / "blocked-check.json"
    preflight.write_text(json.dumps(report), encoding="utf-8")
    result = create_robot_plan(config, tmp_path / "blocked", preflight=preflight)
    assert not result["safe_to_start"]
    assert any("Emergency stop" in item for item in result["blockers"])


def test_robot_check_reports_missing_devices_and_skips(tmp_path):
    config = robot_config(tmp_path)
    report = check_robot(
        config,
        port_provider=lambda: ["COM5"],
        camera_probe=lambda _: (_ for _ in ()).throw(ValueError("no frame")),
        command_finder=lambda name: name if name == "lerobot-record" else None,
    )
    assert not report["ready"]
    failed = {item["code"] for item in report["checks"] if item["status"] == "fail"}
    assert {"SERIAL_PORT", "CAMERA_FRAME", "LEROBOT_COMMAND"} <= failed
    skipped = check_robot(
        config,
        probe_ports=False,
        probe_cameras=False,
        command_finder=lambda _: "available",
    )
    assert skipped["status"] == "incomplete" and not skipped["ready"]
    with pytest.raises(ValueError, match="must not overwrite"):
        write_robot_check(config, config, probe_ports=False, probe_cameras=False)


def test_rollout_evaluation_reports_uncertainty_and_failures(tmp_path):
    source = tmp_path / "rollouts.jsonl"
    rows = [
        {"episode_id": "1", "success": True, "duration_s": 10, "task": "pick"},
        {
            "episode_id": "2",
            "success": False,
            "duration_s": 20,
            "task": "pick",
            "failure_mode": "drop",
            "interventions": 1,
        },
        {"episode_id": "3", "success": True, "duration_s": 12, "task": "place"},
    ]
    source.write_text("\n".join(json.dumps(x) for x in rows), encoding="utf-8")
    result = write_rollout_evaluation(source, tmp_path / "report.json")
    assert result["overall"]["success_rate"] == pytest.approx(2 / 3)
    low, high = result["overall"]["wilson_95_interval"]
    assert 0 <= low < 2 / 3 < high <= 1
    assert result["failure_modes"] == {"drop": 1}
    assert set(result["by_task"]) == {"pick", "place"}


def test_rollout_duplicate_and_empty_rejected(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        evaluate_rollouts(empty)
    duplicate = tmp_path / "duplicate.jsonl"
    row = json.dumps({"episode_id": "same", "success": True, "duration_s": 1})
    duplicate.write_text(row + "\n" + row, encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate_rollouts(duplicate)
    with pytest.raises(ValueError, match="must not overwrite"):
        write_rollout_evaluation(duplicate, duplicate)


def session_inputs(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "protocol_id": "pick-v1",
                "task": "pick",
                "success_definition": (
                    "Object is lifted above the marked line without human contact."
                ),
                "reset_procedure": "Return home and place the object on the marked start position.",
                "environment": "fixed table",
                "required_episodes": 2,
                "max_duration_s": 30,
                "allowed_failure_modes": ["drop"],
            }
        ),
        encoding="utf-8",
    )
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema": "vla-preflight.robot-plan/1",
                "safe_to_start": True,
                "config": {"name": "arm"},
            }
        ),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"policy")
    log = tmp_path / "rollouts.jsonl"
    log.write_text(
        "\n".join(
            json.dumps(item)
            for item in (
                {"episode_id": "a", "success": True, "duration_s": 10, "task": "pick"},
                {
                    "episode_id": "b",
                    "success": False,
                    "duration_s": 12,
                    "task": "pick",
                    "failure_mode": "drop",
                },
            )
        ),
        encoding="utf-8",
    )
    return log, protocol, plan, checkpoint


def test_rollout_session_binds_identity_and_is_comparable(tmp_path):
    log, protocol, plan, checkpoint = session_inputs(tmp_path)
    digest = "a" * 64
    first = create_rollout_session(
        log,
        protocol,
        tmp_path / "session-a",
        robot_plan=plan,
        checkpoint=checkpoint,
        dataset_digest=digest,
    )
    second = create_rollout_session(
        log,
        protocol,
        tmp_path / "session-b",
        robot_plan=plan,
        checkpoint=checkpoint,
        dataset_digest=digest,
    )
    assert first["complete"] and first["identity"]["checkpoint"]["sha256"]
    assert (tmp_path / "session-a/report.html").is_file()
    comparison = compare_rollout_sessions([tmp_path / "session-a", tmp_path / "session-b"])
    assert comparison["comparable"]
    second["identity"]["dataset_digest"] = "b" * 64
    (tmp_path / "session-b/manifest.json").write_text(json.dumps(second), encoding="utf-8")
    assert not compare_rollout_sessions([tmp_path / "session-a", tmp_path / "session-b"])[
        "comparable"
    ]


def test_rollout_session_enforces_protocol(tmp_path):
    log, protocol, plan, checkpoint = session_inputs(tmp_path)
    rows = log.read_text(encoding="utf-8").replace('"task": "pick"', '"task": "place"', 1)
    log.write_text(rows, encoding="utf-8")
    with pytest.raises(ValueError, match="protocol task"):
        create_rollout_session(
            log,
            protocol,
            tmp_path / "bad-session",
            robot_plan=plan,
            checkpoint=checkpoint,
            dataset_digest="a" * 64,
        )


def test_new_cli_workflows_write_artifacts(tmp_path):
    doctor_path = tmp_path / "doctor.json"
    assert main(["doctor", "--output", str(doctor_path)]) == 0
    assert json.loads(doctor_path.read_text())["schema"] == "vla-preflight.doctor/1"

    config = robot_config(tmp_path)
    check_path = tmp_path / "robot-check.json"
    assert (
        main(
            [
                "robot-check",
                str(config),
                "--output",
                str(check_path),
                "--skip-ports",
                "--skip-cameras",
            ]
        )
        == 1
    )
    assert json.loads(check_path.read_text())["status"] in ("failed", "incomplete")

    log, protocol, plan, checkpoint = session_inputs(tmp_path / "session-inputs")
    session = tmp_path / "cli-session"
    assert (
        main(
            [
                "rollout-import",
                str(log),
                "--protocol",
                str(protocol),
                "--robot-plan",
                str(plan),
                "--checkpoint",
                str(checkpoint),
                "--dataset-digest",
                "a" * 64,
                "--output",
                str(session),
            ]
        )
        == 0
    )
    assert main(["rollout-compare", str(session), str(session)]) == 0
