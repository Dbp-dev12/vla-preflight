"""CLI with CI-friendly exit codes: 0 clean, 1 findings, 2 invalid invocation/incomplete."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .audit import audit
from .contract import from_lerobot, load_contract
from .demo import FAULTS, create_demo
from .workflow_cli import dispatch, register


def _episodes(text):
    try:
        ids = {int(x) for x in text.split(",")}
        if not ids or min(ids) < 0:
            raise ValueError
        return ids
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use comma-separated nonnegative episode IDs") from exc


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Local VLA workbench: audit, prepare, train, inspect. Core audit needs no GPU."
    )
    parser.add_argument("--version", action="version", version=__version__)
    subs = parser.add_subparsers(dest="command", required=True)
    scan = subs.add_parser("audit", help="Check a local LeRobot v2.1 or v3.0 dataset")
    scan.add_argument("dataset", type=Path)
    configs = scan.add_mutually_exclusive_group()
    configs.add_argument("--contract", type=Path, help="Explicit audit contract JSON")
    configs.add_argument(
        "--train-config", type=Path, help="Resolved LeRobot train JSON; no custom processors"
    )
    scan.add_argument("--episodes", type=_episodes, help="Audit selected episodes only, e.g. 0,1,4")
    scan.add_argument("--video", choices=["exists", "probe", "skip"], default="exists")
    scan.add_argument(
        "--windows", type=Path, help="Actual sampler trace JSONL (see docs/contracts.md)"
    )
    scan.add_argument("--timestamp-tolerance", type=float, default=0.001, metavar="SECONDS")
    scan.add_argument("--output", type=Path, help="Write report JSON outside dataset")
    scan.add_argument(
        "--html", type=Path, help="Write a self-contained HTML report outside dataset"
    )
    scan.add_argument("--json", action="store_true", help="Emit only JSON to stdout")
    scan.add_argument("--strict", action="store_true", help="Also fail on warnings")
    demo = subs.add_parser("demo", help="Generate a tiny synthetic dataset and contract")
    demo.add_argument("destination", type=Path)
    demo.add_argument("--format", choices=["v2.1", "v3.0"], default="v3.0")
    demo.add_argument("--fault", choices=["none", *FAULTS], default="none")
    register(subs)
    args = parser.parse_args(argv)
    try:
        workflow_result = dispatch(args)
        if workflow_result is not None:
            return workflow_result
        if args.command == "demo":
            create_demo(args.destination, version=args.format, fault=args.fault)
            print(f"Created synthetic fixture: {args.destination}")
            print(
                f'vla-preflight audit "{args.destination / "dataset"}" '
                f'--contract "{args.destination / "contract.json"}" '
                f'--windows "{args.destination / "windows.jsonl"}"'
            )
            return 0
        contract = (
            load_contract(args.contract)
            if args.contract
            else from_lerobot(args.train_config)
            if args.train_config
            else None
        )
        # Validate all destinations before running or writing any reports.
        sources = {p.resolve() for p in (args.contract, args.train_config, args.windows) if p}
        destinations = [p.resolve() for p in (args.output, args.html) if p]
        if len(set(destinations)) != len(destinations):
            raise ValueError("JSON and HTML output paths must differ")
        for path in destinations:
            if path.is_relative_to(args.dataset.resolve()) or path in sources:
                raise ValueError("Reports must not overwrite input files or be inside the dataset")
        if args.output and args.output.suffix.lower() != ".json":
            raise ValueError("--output must end in .json")
        if args.html and args.html.suffix.lower() != ".html":
            raise ValueError("--html must end in .html")
        report = audit(
            args.dataset,
            contract=contract,
            episodes=args.episodes,
            video=args.video,
            windows=args.windows,
            timestamp_tolerance=args.timestamp_tolerance,
        )
        for destination in (args.output, args.html):
            if destination:
                report.write(destination, dataset_root=args.dataset)
        if args.json:
            print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, allow_nan=False))
        else:
            print(f"VLA Preflight {__version__}: {report.status.upper()}")
            print(
                f"Scanned {report.coverage['rows_scanned']} rows, "
                f"{report.coverage['files_scanned']} files. "
                f"Scope: {report.coverage.get('scope', 'unresolved')}."
            )
            for finding in report.findings.values():
                print(
                    f"[{finding.severity.upper()}] {finding.code} ({finding.count}): "
                    f"{finding.message}"
                )
                print(f"  Next: {finding.suggestion}")
                if finding.examples:
                    print(f"  Example: {json.dumps(finding.examples[0], ensure_ascii=False)}")
            print("Coverage: " + json.dumps(report.coverage, ensure_ascii=False))
            print("No model/robot was run. A pass applies only to listed checks and scanned scope.")
        if not report.complete:
            return 2
        return 1 if report.status == "failed" or (args.strict and report.status == "warning") else 0
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"vla-preflight: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
