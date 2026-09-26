# VLA Preflight Workbench

**Inspect robot-learning data, prepare reproducible splits, and run a small VLA experiment locally.**

[中文](README.zh-CN.md) · [Workflow guide](docs/workflows.md) · [Validation](docs/validation.md) · [Audit checks](docs/checks.md)

v0.2.0 alpha connects data diagnostics to actual training artifacts. It targets people with a laptop, local LeRobot v2.1/v3.0 data and no robot. The local web workbench needs no Node build step or cloud account.

| Capability | What actually happens |
|---|---|
| Inspect | Audit schema, numeric trajectories, statistics, sampler traces and video references; view images and action curves |
| Prepare | Group exact numeric duplicates, split whole episodes, fit **train-only** normalization, fingerprint inputs |
| Train | Run a randomly initialized image + byte-text + state model with masked action chunks and held-out evaluation |
| Compare | Record baselines, loss curves, last/best checkpoints, sampler traces and compatible resumed runs |
| Export | Filter episodes into a new v3 dataset, reindex and recompute numeric statistics |
| Extend | Generate an **experimental** SmolVLA plan with a physically separate training split |

Tiny VLA is a reference trainer, **not a pretrained foundation model**. Offline action error is not robot task success. SmolVLA execution and memory use have not been validated. This project does not claim novelty over every GitHub repository or replace LeRobot.

## Start locally

Use Python 3.10–3.13. Create and activate an environment:

```shell
python -m venv .venv
# Windows: .venv/Scripts/activate
# Linux/macOS: source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[train]"
vla-preflight learning-demo demo-output/learning
vla-preflight studio demo-output/learning --workspace workbench-output/learning --open
```

Open `http://127.0.0.1:8765`. Inspect episodes → create preparation bundle → select it → train. The default 24-episode synthetic set can overfit: compare with the constant-action baselines before interpreting decreasing training loss.

After installation, Windows users can run `Run-Workbench.cmd` to generate and open the demo. CPU wheels avoid unnecessary CUDA downloads. `uv sync --extra dev --locked` installs core/development dependencies; the lockfile's optional train extra uses the default package index and may bring large accelerator dependencies on Linux.

## CLI workflow

```shell
vla-preflight profile demo-output/learning --output reports/profile.json
vla-preflight prepare demo-output/learning --output runs/prepared
vla-preflight train runs/prepared --output runs/first --steps 200
vla-preflight train runs/prepared --output runs/resumed --steps 400 --resume runs/first/checkpoint.pt
vla-preflight compare runs/first runs/resumed --output reports/comparison.json
vla-preflight export demo-output/learning --exclude 0,1 --output datasets/filtered
```

Outputs must be outside source data. Preparation, training, export and plan destinations must be new directories. Resume uses a new run directory; `--steps` is the **total target**. Source fingerprints and preparation artifacts are checked before training.

Only need preflight? `pip install -e .` installs no PyTorch/imaging dependencies:

```shell
vla-preflight audit /path/to/dataset --output reports/audit.json --html reports/audit.html
vla-preflight demo demo-output/broken --fault cross-episode
vla-preflight audit demo-output/broken/dataset --contract demo-output/broken/contract.json --windows demo-output/broken/windows.jsonl
```

The final command intentionally exits 1 and detects `WINDOW_CROSS_EPISODE`. Audit exit codes: 0=no errors, 1=findings/strict warnings, 2=incomplete input/scan. See [contracts](docs/contracts.md) for declared assumptions versus physical truth.

## Experimental pretrained-model bridge

```shell
vla-preflight smol-plan runs/prepared --output runs/smol-plan --python /path/to/lerobot-env/bin/python
# Explicit launch: may download weights and use your GPU.
vla-preflight smol-launch runs/smol-plan
```

Use a separate environment compatible with [LeRobot](https://github.com/huggingface/lerobot) and [SmolVLA](https://huggingface.co/docs/lerobot/smolvla). The bridge does not install LeRobot, guarantee upstream format/API compatibility, provide external-model evaluation, or claim measured memory requirements. Opening the workbench or generating a plan never downloads a model.

## Scope and adoption

The useful contribution is a connected workflow: the same preparation bundle binds split, statistics, fingerprints, actual sampled chunks and training results. Existing robotics tools cover many individual pieces. We welcome small reproducible datasets and compatibility reports.

- Local Parquet layouts only; no RLDS/Lance connector or automatic dataset downloads in the workbench.
- Audit streams numeric rows. Profiling retains numeric rows in RAM. Training preloads selected-camera 32×32 images, up to 50,000 frames by default. Export subsets for larger data.
- Duplicate groups compare action/state and episode task labels, not images or semantics; no task-stratified split.
- Lag correlation is exploratory, not a latency measurement or automatic repair.
- Reference model uses one camera, state, first 64 UTF-8 instruction bytes and action chunks; no robot deployment.
- Sources are read-only. Exports may copy entire shared videos including excluded footage. Review paths/task text/evidence before sharing.
- Local server binds only `127.0.0.1`. No telemetry/uploads. Not a multiuser service.

## Development

```shell
python -m pip install -e ".[dev,train]"
python -m pytest
ruff check .
ruff format --check .
python scripts/fault_matrix.py --output reports/fault-matrix.json
python -m build
```

Training tests skip without optional dependencies; install `[dev,train]` for the full suite. [CONTRIBUTING.md](CONTRIBUTING.md) explains useful bug reports. MIT covers original code; datasets/models retain their own licenses. See [validation](docs/validation.md) for tested and untested boundaries.
