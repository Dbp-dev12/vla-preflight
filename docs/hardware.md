# GPU and robot workflow

VLA Preflight v0.6 adds evidence and gates around a physical workflow while leaving motor control to LeRobot. A configuration file cannot prove that an emergency stop works, calibration is correct or a workspace is safe.

## 1. Inspect the training machine

```shell
vla-preflight doctor --output reports/doctor.json
```

The report includes Python, operating system family, installed package versions, command availability, CUDA device names and VRAM. It intentionally omits hostnames, usernames, environment variables and absolute paths. Memory guidance is a starting heuristic, not a measured SmolVLA requirement.

## 2. Declare the cell

Copy `examples/robot.example.json` and edit the robot, teleoperator, camera, dataset and task fields. Keep the safety gates false until they have been physically checked. Calibration paths are resolved relative to the config file and their SHA-256 digests are recorded.

```shell
vla-preflight robot-plan robot.json --output runs/robot-plan
```

Exit code 1 means a safety or calibration gate is blocked. Exit code 0 means the declarations allow supervised low-speed bring-up; it does not certify the hardware. The generated `RUNBOOK.md` preserves an ordered review process and `plan.json` stores commands as argument arrays to avoid shell ambiguity.

Command shapes follow the current official [LeRobot recording script](https://github.com/huggingface/lerobot/blob/main/src/lerobot/scripts/lerobot_record.py) and [teleoperation script](https://github.com/huggingface/lerobot/blob/main/src/lerobot/scripts/lerobot_teleoperate.py). Review them against the installed LeRobot version before execution.

## 3. Collect and audit

Run supervised teleoperation first, then record a small disposable dataset. Audit that dataset before scaling collection:

```shell
vla-preflight audit /path/to/dataset --video probe --output reports/audit.json
vla-preflight prepare /path/to/dataset --output runs/prepared
```

Keep a held-out evaluation protocol, object layout and reset procedure fixed before comparing checkpoints. Dataset fingerprints help detect local changes but are not signed provenance.

## 4. Record task-level outcomes

Write one JSON object per physical or simulated episode:

```json
{"episode_id":"eval-001","success":true,"duration_s":12.4,"interventions":0,"task":"red block to bowl","checkpoint":"policy-10000","dataset_digest":"..."}
```

Then summarize it:

```shell
vla-preflight rollout-eval rollouts.jsonl --output reports/rollouts.json
```

The report includes the observed success rate and a Wilson 95% interval. It also separates tasks and checkpoints and counts human interventions and failure modes. The interval captures binomial sampling uncertainty only; it does not correct for operator bias, task selection, resets or changing physical conditions.
