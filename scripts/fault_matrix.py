"""Run all generated faults in both layouts. Not an independent accuracy benchmark."""

import argparse
import json
import tempfile
from pathlib import Path

from vla_preflight.audit import audit
from vla_preflight.contract import load_contract
from vla_preflight.demo import FAULTS, create_demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = []
    with tempfile.TemporaryDirectory(prefix="vla-preflight-") as temp:
        for version in ("v2.1", "v3.0"):
            for fault in ("none", *FAULTS):
                root = create_demo(Path(temp) / version / fault, version=version, fault=fault)
                report = audit(
                    root / "dataset",
                    contract=load_contract(root / "contract.json"),
                    windows=root / "windows.jsonl",
                )
                expected = FAULTS.get(fault)
                ok = expected in report.findings if expected else report.status == "passed"
                results.append(
                    {
                        "format": version,
                        "fault": fault,
                        "expected": expected,
                        "observed": sorted(report.findings),
                        "matched": ok,
                    }
                )
    obj = {
        "description": "Synthetic fault regression matrix, not real-world detection accuracy",
        "cases": results,
        "matched": sum(r["matched"] for r in results),
        "total": len(results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    print(f"{obj['matched']}/{obj['total']} synthetic cases matched expectations")
    return 0 if all(r["matched"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
