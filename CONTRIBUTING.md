# Contributing

This is an early, independent project. Small reproducible checks are more valuable
than broad claims about model quality. Open an issue before a large API change.

```shell
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
python -m pytest --cov=vla_preflight --cov-report=term-missing
python scripts/fault_matrix.py --output reports/fault-matrix.json
python -m build
```

A new check should include:

1. A real failure description or a clearly labeled synthetic counterexample.
2. A stable finding code, actionable evidence, and a suggested next step.
3. A positive test and a benign counterexample to avoid false alarms.
4. An update to `docs/checks.md` and any scope limitations.

Never modify the input dataset or silently download data in the audit path. Keep
PyTorch/CUDA optional by avoiding them entirely in the core. No credentials, datasets,
large weights or private reports belong in git. Review evidence before attaching it
to an issue. Changes to the contract must update `examples/contract.schema.json`.

Install `.[dev,train,robot]` for the full workflow, training and physical-probe environment.
Robot unit tests use fakes and do not access hardware. Manual probe reports must remain local.
Use CPU PyTorch wheels when CUDA is unnecessary. Preserve train/validation separation,
source immutability and honest baseline reporting when extending the trainer.

Public smoke testing is optional and network-dependent; it is not part of unit CI.
Downloaded data keep their upstream license and must not be added to this repository.

