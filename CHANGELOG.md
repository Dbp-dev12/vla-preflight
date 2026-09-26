# Changelog

## 0.6.0 — 2026-09-26

- Non-actuating robot preflight: serial enumeration, three-frame camera probes, calibration hashes, LeRobot command discovery and explicit safety/coverage gates.
- Robot plans require a passing preflight report bound to the exact configuration digest.
- External LeRobot 0.6.x/Python 3.12/CUDA compatibility reports are required before SmolVLA planning.
- Low-memory plans expose gradient accumulation and checkpointing; execution remains delegated to LeRobot.
- External runs retain data/environment identity, exit status, checkpoint hashes and logged peak memory.
- First-class rollout sessions bind protocol, robot plan, dataset and checkpoint identities and prevent invalid comparisons.
- The local workbench indexes compute, robot, external-training and rollout evidence in a fifth workflow page.
- Versioned examples, acceptance record, hardware runbook and regression coverage for the full workflow.

Still alpha: motor control remains external and physical checks require supervised validation.

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

