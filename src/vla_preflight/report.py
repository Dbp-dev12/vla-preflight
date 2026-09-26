"""Bounded evidence collection and portable JSON/HTML reports."""

from __future__ import annotations

import html
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import __version__


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


@dataclass
class Finding:
    code: str
    severity: str
    message: str
    suggestion: str
    count: int = 0
    examples: list[dict] = field(default_factory=list)


class Report:
    def __init__(self) -> None:
        self.findings: dict[str, Finding] = {}
        self.coverage: dict = {"rows_scanned": 0, "files_scanned": 0}
        self.complete = True

    def add(self, code, severity, message, suggestion, **evidence):
        finding = self.findings.setdefault(code, Finding(code, severity, message, suggestion))
        finding.count += 1
        if len(finding.examples) < 5:
            finding.examples.append(_json_safe(evidence))

    @property
    def status(self):
        if any(f.severity == "error" for f in self.findings.values()):
            return "failed"
        if not self.complete:
            return "incomplete"
        if any(f.severity == "warning" for f in self.findings.values()):
            return "warning"
        return "passed"

    def to_dict(self):
        return {
            "schema_version": "1.0",
            "tool_version": __version__,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": self.status,
            "scan_complete": self.complete,
            "coverage": self.coverage,
            "findings": [asdict(f) for f in self.findings.values()],
            "limitations": [
                "Passing means only that the listed checks found no issue in the scanned scope.",
                "No model training, robot execution, or task-success evaluation is performed.",
                "Physical units, coordinate frames and sensor alignment are not inferred.",
                "Video probing does not verify every frame or semantic synchronization.",
            ],
        }

    def html(self):
        data = self.to_dict()
        esc = lambda value: html.escape(str(value))  # noqa: E731
        cards = "".join(
            f'<article class="{esc(f.severity)}"><h2>{esc(f.code)} '
            f"<small>{esc(f.severity)} · {f.count} occurrence(s)</small></h2>"
            f"<p>{esc(f.message)}</p><p><b>Next:</b> {esc(f.suggestion)}</p>"
            f"<pre>{esc(json.dumps(f.examples, indent=2, ensure_ascii=False))}</pre></article>"
            for f in self.findings.values()
        )
        return f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>VLA Preflight report</title><style>
body{{font:16px/1.6 system-ui,sans-serif;background:#f5f7fb;color:#17243b;
max-width:1000px;margin:auto;padding:28px}}h1{{margin-bottom:0}}small{{font-size:13px}}
article{{background:white;padding:18px 24px;margin:18px 0;border-left:5px solid #527a99;
border-radius:8px}}.error{{border-color:#bb3434}}.warning{{border-color:#b47a12}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#eef2f7;padding:14px}}
h2{{font-size:19px}}footer{{color:#516079}}</style>
<h1>VLA Preflight</h1><p>v{__version__} · <b>{esc(self.status.upper())}</b>
· scan complete: {self.complete}</p><h2>What was checked</h2>
<pre>{esc(json.dumps(self.coverage, indent=2, ensure_ascii=False))}</pre>
{cards or "<article>No findings in the scanned scope.</article>"}
<footer><h2>Scope limits</h2><ul>
{"".join("<li>" + esc(x) + "</li>" for x in data["limitations"])}
</ul><p>Local, static report. No external scripts, fonts, or telemetry.</p></footer></html>"""

    def write(self, path: Path, *, dataset_root: Path):
        path = path.resolve()
        if path.is_relative_to(dataset_root.resolve()):
            raise ValueError("Report must be outside the source dataset (read-only guarantee).")
        path.parent.mkdir(parents=True, exist_ok=True)
        content = (
            self.html()
            if path.suffix.lower() == ".html"
            else json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False)
        )
        path.write_text(content + "\n", encoding="utf-8")
