import http.client
import json
import threading

import numpy as np
import pytest

from vla_preflight.analysis import Moments, prepare, split_episodes
from vla_preflight.contract import load_json
from vla_preflight.dataset import Dataset
from vla_preflight.export import export_episodes
from vla_preflight.workflow_io import atomic_json, episode_records, fingerprint, load_prepared


@pytest.fixture
def visual(tmp_path):
    pytest.importorskip("PIL")
    from vla_preflight.learning_demo import learning_demo

    return learning_demo(tmp_path / "source", episodes=8, frames=8)


def test_moments_matches_numpy():
    x = np.random.default_rng(31).normal(size=(137, 3))
    moment = Moments(3)
    for row in x:
        moment.add(row)
    result = moment.result()
    np.testing.assert_allclose(result["mean"], x.mean(0), atol=1e-14)
    np.testing.assert_allclose(result["std"], x.std(0), atol=1e-14)
    np.testing.assert_allclose(result["q01"], np.quantile(x, 0.01, axis=0))


def test_duplicate_groups_never_cross_split():
    profile = {"episodes": [{"episode": i, "numeric_hash": str(i // 2)} for i in range(12)]}
    for seed in range(10):
        split = split_episodes(profile, seed=seed)
        assert set(split["train"]).isdisjoint(split["validation"])
        for i in range(0, 12, 2):
            assert (i in split["train"]) == (i + 1 in split["train"])
        assert split == split_episodes(profile, seed=seed)


def test_prepare_fits_train_only_and_detects_mutation(visual, tmp_path):
    target = tmp_path / "prepared"
    prepare(visual, target)
    ds, _, split, normal = load_prepared(target)
    actions = np.asarray([r["action"] for ep in split["train"] for r in episode_records(ds, ep)])
    np.testing.assert_allclose(normal["features"]["action"]["mean"], actions.mean(0))
    assert normal["fit_episodes"] == split["train"]
    path = target / "split.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="artifact changed"):
        load_prepared(target)


def test_source_mutation_rejected(visual, tmp_path):
    target = tmp_path / "prepared"
    prepare(visual, target)
    path = visual / "meta/info.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="Source changed"):
        load_prepared(target)


def test_export_preserves_source_and_images(visual, tmp_path):
    ds = Dataset(visual)
    before = fingerprint(ds)
    target = tmp_path / "exported"
    result = export_episodes(visual, target, exclude={1, 3, 5})
    assert result["complete"]
    assert fingerprint(ds) == before
    copied = Dataset(target)
    assert len(copied.episodes) == 5
    for old, new in result["episode_mapping"].items():
        original, exported = episode_records(ds, int(old)), episode_records(copied, new)
        assert [r["action"] for r in original] == [r["action"] for r in exported]
        assert original[0]["observation.image"] == exported[0]["observation.image"]
        assert {r["episode_index"] for r in exported} == {new}
    with pytest.raises(ValueError, match="outside"):
        export_episodes(visual, visual / "child", exclude=set())
    with pytest.raises(ValueError, match="Unknown"):
        export_episodes(visual, tmp_path / "bad", exclude={999})


def test_smol_plan_is_train_only_without_execution(visual, tmp_path, monkeypatch):
    from vla_preflight.bridge import SmolOptions, create_smol_plan, launch_smol

    prepared = tmp_path / "prepared"
    prepare(visual, prepared)
    environment = tmp_path / "environment.json"
    environment.write_text(
        json.dumps(
            {
                "schema": "vla-preflight.lerobot-environment/1",
                "compatible": True,
                "requested_python": "python-for-test",
                "requested_device": "cuda",
                "observed": {"lerobot_version": "0.6.2"},
            }
        ),
        encoding="utf-8",
    )
    result = create_smol_plan(
        prepared,
        tmp_path / "plan",
        options=SmolOptions(python="python-for-test"),
        environment_report=environment,
    )
    split = load_json(prepared / "split.json")
    export = load_json(tmp_path / "plan/train-dataset/export-status.json")
    assert set(map(int, export["episode_mapping"])) == set(split["train"])
    assert result["status"] == "planned"
    assert result["effective_batch_size"] == 8
    assert "--accelerator.gradient_accumulation.steps=8" in result["command_preview"]
    assert "--policy.gradient_checkpointing=true" in result["command_preview"]
    assert not (tmp_path / "plan/artifacts").exists()

    class Process:
        def __init__(self, command, cwd, stdout, stderr):
            artifacts = tmp_path / "plan/artifacts/checkpoints/last"
            artifacts.mkdir(parents=True)
            (artifacts / "model.safetensors").write_bytes(b"weights")
            stdout.write("peak memory: 2048 MiB\n")
            stdout.flush()

        def wait(self, timeout=None):
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr("vla_preflight.bridge.subprocess.Popen", Process)
    completed = launch_smol(tmp_path / "plan")
    assert completed["status"] == "completed"
    assert completed["measured_peak_memory_gib"] == 2
    external = load_json(tmp_path / "plan/external-run.json")
    assert external["checkpoint_files"][0]["sha256"]


def test_http_host_origin_token_and_actual_job(visual, tmp_path):
    from vla_preflight.studio import Workspace, make_server

    workspace = Workspace(visual, tmp_path / "studio")
    server = make_server(workspace, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host = f"127.0.0.1:{server.server_port}"

    def request(method, path, data=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port)
        connection.request(method, path, json.dumps(data) if data else None, headers or {})
        response = connection.getresponse()
        status, body = response.status, response.read()
        connection.close()
        return status, body

    try:
        status, body = request("GET", "/api/state")
        assert status == 200 and "evidence" in json.loads(body)
        status, body = request("GET", "/")
        assert status == 200 and "硬件与实机" in body.decode()
        assert request("GET", "/", headers={"Host": "evil.example"})[0] == 403
        assert request("POST", "/api/jobs", {"kind": "analyze"})[0] == 403
        headers = {"Origin": f"http://{host}", "X-Preflight-Token": workspace.token}
        status, body = request("POST", "/api/jobs", {"kind": "prepare"}, headers)
        assert status == 202
        job_id = json.loads(body)["id"]
        workspace.pool.shutdown(wait=True)
        assert workspace.jobs[job_id]["status"] == "completed"
        assert len(workspace.snapshot()["prepared"]) == 1
        status, body = request("GET", "/api/image?episode=0&frame=0")
        assert status == 200 and body.startswith(b"\x89PNG")
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        workspace.close()


def test_workspace_indexes_hardware_and_rollout_evidence(visual, tmp_path):
    from vla_preflight.studio import JobRequest, Workspace

    workspace = Workspace(visual, tmp_path / "studio-evidence")
    job_id = workspace.start(JobRequest(kind="doctor"))
    workspace.pool.shutdown(wait=True)
    try:
        snapshot = workspace.snapshot()
        assert workspace.jobs[job_id]["status"] == "completed"
        assert len(snapshot["evidence"]["doctor"]) == 1
        atomic_json(
            workspace.output / "robot-checks/check.json",
            {"status": "passed", "ready": True, "checks": [{"status": "pass"}]},
        )
        atomic_json(
            workspace.output / "robot-plans/plan/plan.json",
            {
                "status": "ready-for-supervised-bringup",
                "safe_to_start": True,
                "config": {"name": "arm"},
            },
        )
        atomic_json(
            workspace.output / "rollouts/session/manifest.json",
            {
                "status": "complete",
                "protocol": {"protocol_id": "p1"},
                "episodes": 20,
                "required_episodes": 20,
                "summary": {"overall": {"success_rate": 0.5}},
            },
        )
        atomic_json(
            workspace.output / "environment/env.json",
            {
                "status": "compatible",
                "observed": {"lerobot_version": "0.6.2", "cuda_available": True},
            },
        )
        atomic_json(
            workspace.output / "smol-plans/run/external-run.json",
            {
                "status": "completed",
                "checkpoint_files": [{"path": "model.safetensors"}],
                "measured_peak_memory_gib": 5.5,
            },
        )
        evidence = workspace.snapshot()["evidence"]
        assert evidence["robot_checks"][0]["ready"]
        assert evidence["robot_plans"][0]["safe_to_start"]
        assert evidence["rollout_sessions"][0]["protocol"] == "p1"
        assert evidence["lerobot_environments"][0]["version"] == "0.6.2"
        assert evidence["external_runs"][0]["checkpoints"] == 1
    finally:
        workspace.close()
