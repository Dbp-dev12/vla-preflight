# v0.3 → v0.6 acceptance record

The early v0.6 tag was a preview. The release is publishable only when all acceptance items below pass together.

## v0.3 — Robot preflight

- Strict robot, teleoperator, camera, calibration and safety schema.
- Enumerates declared serial ports without opening them.
- Opens each declared camera, reads three frames and checks dimensions/nonconstant pixels.
- Verifies `lerobot-teleoperate` and `lerobot-record` are discoverable.
- Records skipped coverage as incomplete; never treats skipped checks as passed.
- A robot plan requires a passing report bound to the same config digest.

## v0.4 — External LeRobot GPU training

- Training remains inside a separate LeRobot environment; VLA Preflight contains no replacement foundation-model trainer.
- Checks Python 3.12+, LeRobot 0.6.x, train module, SmolVLA module and requested CUDA availability.
- A SmolVLA plan requires the compatible environment report.
- External runs bind the training-data and environment digests and hash every safetensors checkpoint.
- Exit code, duration and logged peak memory are retained as external-run evidence.

## v0.5 — First-class physical outcomes

- Strict evaluation protocol fixes task, success definition, reset, environment, episode count and timeout.
- Session identity binds protocol, robot plan, dataset SHA-256 and checkpoint SHA-256.
- Normalized episodes, manifest, summary and offline HTML are retained together.
- Incomplete episode counts remain incomplete.
- Direct comparison is allowed only when protocol, robot plan and dataset identity match.

## v0.6 — Integrated workbench

- The loopback workbench runs compute, robot-check, robot-plan and rollout-import jobs.
- Hardware and rollout evidence is indexed in `/api/state` and visible in the fifth UI workflow page.
- Existing data audit, preparation, Tiny VLA and external LeRobot paths remain available.
- Unit/integration tests, fault matrix, JavaScript syntax, wheel build and installed-wheel smoke must all pass.
