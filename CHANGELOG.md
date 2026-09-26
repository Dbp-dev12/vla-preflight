# Changelog

## 0.6.0 — 2026-09-26

- Privacy-conscious `doctor` report for CUDA devices, VRAM, packages and LeRobot command availability.
- Low-memory SmolVLA plans now expose gradient accumulation and gradient checkpointing.
- Validated robot/camera/teleoperator declarations with calibration hashes and four explicit safety gates.
- Reviewable `lerobot-teleoperate` and `lerobot-record` command generation; the tool never opens hardware.
- Rollout JSONL evaluation with task/checkpoint slices, interventions, failure modes and Wilson intervals.
- Hardware runbook, safe blocked example and regression coverage for the new workflow.

Still alpha: physical control and pretrained-model execution remain external and require supervised validation.

## 0.2.0 — 2026-09-26

- Local browser workbench with real background preparation, export and training jobs.
- Numeric profiles, duplicate-group episode splits, train-only statistics and fingerprints.
- Synthetic visual/language/action data and a real PyTorch reference VLA trainer.
- Masked action chunks, baseline-aware evaluation, atomic latest/best checkpoints and resume.
- Filtered v3 export with provenance and post-export verification.
- Experimental external SmolVLA plans; pretrained-model execution remains unvalidated.
- Regression tests for actual gradients, CPU resume, split isolation and HTTP jobs.

Still alpha: no robot task success or measured pretrained-model resource claim.

## 0.1.0 — 2026-09-25

Initial alpha release:

- Read-only numeric audits for LeRobot v2.1 and v3.0 Parquet layouts.
- Explicit audit contracts and a restricted resolved LeRobot train-config reader.
- Stored action/state statistics validation, split and declared-semantic checks.
- Optional exported sampler trace checks and video existence/ffprobe checks.
- CLI, Python API, JSON and offline HTML reports with bounded evidence and coverage.
- Deterministic fault fixtures, regression tests and a pinned public numeric smoke script.

No model training, actual preprocessing replay or robot execution is implemented.

