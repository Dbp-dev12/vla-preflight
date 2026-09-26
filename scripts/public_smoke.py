"""Explicit opt-in download of a pinned public numeric dataset (no videos or weights)."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from vla_preflight.audit import audit
from vla_preflight.contract import Contract

REPO = "lerobot/pusht"
REVISION = "7628202a2180972f291ba1bc6723834921e72c19"
FILES = [
    "meta/info.json",
    "meta/stats.json",
    "meta/tasks.parquet",
    "meta/episodes/chunk-000/file-000.parquet",
    "data/chunk-000/file-000.parquet",
]
MAX_FILE_BYTES = 50 * 1024 * 1024


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.destination.exists():
        parser.error("Destination must be new; this script never overwrites a dataset")
    if args.report.resolve().is_relative_to(args.destination.resolve()):
        parser.error("Report must be outside downloaded dataset")
    manifest = {}
    for relative in FILES:
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{relative}"
        print(f"Downloading {relative}", flush=True)
        with urllib.request.urlopen(url, timeout=60) as response:
            content = response.read(MAX_FILE_BYTES + 1)
        if len(content) > MAX_FILE_BYTES:
            raise ValueError(f"File exceeds 50 MiB limit: {relative}")
        path = args.destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        manifest[relative] = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    report = audit(
        args.destination,
        video="skip",
        contract=Contract(
            action_dim=2,
            state_dim=2,
            fps=10,
            camera_keys=["observation.image"],
            normalization={"action": "MEAN_STD", "observation.state": "MEAN_STD"},
        ),
    )
    result = report.to_dict()
    result["provenance"] = {
        "repo_id": REPO,
        "revision": REVISION,
        "files": manifest,
        "scope": "numeric-only; videos not downloaded; no model evaluation",
        "dataset_license": "Check the upstream dataset card; data are not redistributed",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{report.status}: {report.coverage['rows_scanned']} rows scanned")
    return 0 if report.complete and report.status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
