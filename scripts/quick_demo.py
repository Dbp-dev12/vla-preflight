"""Generate clean/fault reports in a new directory. Open a report only with --open."""

import argparse
import tempfile
import webbrowser
from pathlib import Path

from vla_preflight.audit import audit
from vla_preflight.contract import load_contract
from vla_preflight.demo import create_demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--open", action="store_true", help="Open the local fault report in a browser"
    )
    args = parser.parse_args()
    output = Path("demo-output")
    output.mkdir(exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="run-", dir=output))
    for name, fault in (("clean", "none"), ("broken", "cross-episode")):
        dataset = create_demo(root / name, fault=fault)
        report = audit(
            dataset / "dataset",
            contract=load_contract(dataset / "contract.json"),
            windows=dataset / "windows.jsonl",
        )
        report.write(root / f"{name}.html", dataset_root=dataset / "dataset")
        report.write(root / f"{name}.json", dataset_root=dataset / "dataset")
        print(f"{name}: {report.status} (expected {'passed' if name == 'clean' else 'failed'})")
    print(f"Reports: {root.resolve()}")
    if args.open:
        webbrowser.open((root / "broken.html").resolve().as_uri())


if __name__ == "__main__":
    main()
