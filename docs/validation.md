# v0.2.0 validation record

Validation date: **2026-09-26**. Verification used CPU-only PyTorch with the
optional image and video dependencies. Host paths, account details, hardware
identifiers and other machine-specific information are intentionally omitted.

| Check | Observed result |
|---|---|
| Regression suite | 111 tests passed |
| Synthetic fault matrix | 20/20 expected outcomes |
| Real gradient updates | Image, text, state and prediction-head weights changed |
| CPU resume | Exact parameter equality versus uninterrupted training |
| Preparation | Train-only statistics, grouped splits and mutation rejection tested |
| Export | Source unchanged; image/action contents preserved with remapped IDs |
| Video decoding | Encoded test clip decoded with a nonzero episode start offset |
| Workbench | Preview, preparation, training, comparison and export exercised |
| Public numeric data | Pinned PushT revision: 206 episodes and 25,650 frames |
| Installed wheel | Independent install, web routes and training smoke passed |
| Lint, format and build | Passed |

Workbench testing found and fixed delayed saving of numeric form values. No console
errors were observed in the exercised flows. These are manual browser checks, not an
automated browser test suite. Public-data video checks were intentionally skipped.

## Actual training outcome

The generated reaching task is synthetic and is not robot demonstration data. A
120-episode, 16-frame-per-episode experiment used seed 7, a 96/24 episode split,
400 steps, batch size 32 and learning rate 0.0003:

- Initial held-out MAE: **0.0146476254**.
- Final held-out MAE: **0.0195798371**.
- Training-mean-action baseline MAE: **0.0146593004**.
- Best evaluated candidate: initialization, step 0.

This is an unsuccessful generalization result, deliberately retained. The tool flags
worsening validation and failure to beat constant baselines. It validates the reporting
workflow, not model quality. Best-checkpoint selection uses validation, not an independent test.

## Reproduce

```shell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev,train]"
python -m pytest
ruff check .
ruff format --check .
python scripts/fault_matrix.py --output reports/fault-matrix.json
python -m build
```

To reproduce the larger synthetic experiment in new directories:

```shell
vla-preflight learning-demo demo-output/toy120 --episodes 120
vla-preflight prepare demo-output/toy120 --output runs/toy120-prepared
vla-preflight train runs/toy120-prepared --output runs/toy120-run --steps 400 --batch-size 32 --learning-rate 0.0003
```

The optional public smoke script downloads only pinned metadata and numeric data. It
does not redistribute the dataset, download model weights or evaluate a policy. Source
revision and content hashes are recorded in [public-pusht.json](evidence/public-pusht.json).

## Not established

- SmolVLA or another pretrained-model training run.
- Upstream-loader compatibility of exported datasets.
- External-model held-out evaluation, memory requirements or CUDA reproducibility.
- Robot task success, simulator rollout performance or real-data generalization.
- Large-corpus memory/performance or semantic duplicate detection.
- Remote CI results until the corresponding workflow has completed.

This alpha release is intended for public testing, not production certification.
Generated fault cases are regression evidence, not independent sensitivity or
specificity measurements.
