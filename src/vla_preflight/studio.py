"""Loopback-only workbench, bounded background jobs and explicit artifact directories."""

from __future__ import annotations

import copy
import io
import json
import secrets
import threading
import uuid
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from pydantic import BaseModel, ConfigDict, Field, StrictInt

from . import __version__
from .analysis import prepare, profile_dataset
from .audit import audit
from .contract import load_json
from .dataset import Dataset, inside
from .media import read_image, visual_keys
from .workflow_io import atomic_json, episode_records


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: str
    validation_fraction: float = Field(default=0.2, gt=0, lt=1, allow_inf_nan=False)
    seed: StrictInt = Field(default=7, ge=0, le=2**31 - 1)
    prepared_id: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9_-]{1,80}$")
    resume_id: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9_-]{1,80}$")
    steps: StrictInt = Field(default=200, ge=1, le=100000)
    batch_size: StrictInt = Field(default=16, ge=1, le=512)
    learning_rate: float = Field(default=0.001, gt=0, le=0.1, allow_inf_nan=False)
    chunk_size: StrictInt = Field(default=4, ge=1, le=64)
    exclude: list[StrictInt] = Field(default_factory=list)
    python: str | None = None


class Workspace:
    def __init__(self, root: Path, output: Path):
        self.dataset = Dataset(root)
        self.root, self.output = self.dataset.root, output.resolve()
        if self.output.is_relative_to(self.root):
            raise ValueError("Workbench workspace must be outside source dataset")
        context = self.output / "context.json"
        if context.exists() and load_json(context).get("dataset_root") != str(self.root):
            raise ValueError(
                "This workspace belongs to a different dataset; choose another workspace"
            )
        atomic_json(context, {"dataset_root": str(self.root)})
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="preflight")
        self.jobs, self.cancellations = {}, {}
        for path in (self.output / "jobs").glob("*.json"):
            job = load_json(path)
            if job["status"] in ("running", "queued"):
                job["status"] = "interrupted"
            self.jobs[job["id"]] = job

    def _update(self, job_id, **values):
        with self.lock:
            self.jobs[job_id].update(values)
            atomic_json(self.output / "jobs" / f"{job_id}.json", self.jobs[job_id])

    def start(self, request: JobRequest):
        if request.kind not in ("analyze", "prepare", "export", "train", "smol-plan"):
            raise ValueError("Unsupported workbench job")
        with self.lock:
            if any(j["status"] in ("queued", "running") for j in self.jobs.values()):
                raise ValueError("A job is already active; wait for it or cancel a training job")
            job_id = uuid.uuid4().hex[:12]
            self.jobs[job_id] = {
                "id": job_id,
                "kind": request.kind,
                "status": "queued",
                "metrics": [],
            }
            self.cancellations[job_id] = threading.Event()
        self.pool.submit(self._execute, job_id, request)
        return job_id

    def _execute(self, job_id, req):
        self._update(job_id, status="running")
        try:
            if req.kind == "analyze":
                report = audit(self.root)
                result = {"audit": report.to_dict(), "profile": profile_dataset(self.root)}
                atomic_json(self.output / "analysis.json", result)
                artifact = "analysis.json"
            elif req.kind == "prepare":
                target = self.output / "prepared" / job_id
                prepare(
                    self.root, target, validation_fraction=req.validation_fraction, seed=req.seed
                )
                artifact = f"prepared/{job_id}"
            elif req.kind == "export":
                from .export import export_episodes

                target = self.output / "exports" / job_id
                export_episodes(self.root, target, exclude=set(req.exclude))
                artifact = f"exports/{job_id}"
            elif req.kind == "train":
                from .training import TrainOptions, train

                prepared = self._prepared(req.prepared_id)
                target = self.output / "runs" / job_id
                resume = None
                if req.resume_id:
                    resume = inside(self.output, f"runs/{req.resume_id}/checkpoint.pt")
                    if not resume.is_file():
                        raise ValueError("Resume checkpoint is unavailable")

                def progress(item):
                    with self.lock:
                        values = (self.jobs[job_id]["metrics"] + [item])[-500:]
                    self._update(job_id, metrics=values)

                result = train(
                    prepared,
                    target,
                    resume=resume,
                    options=TrainOptions(
                        steps=req.steps,
                        batch_size=req.batch_size,
                        learning_rate=req.learning_rate,
                        seed=req.seed,
                        chunk_size=req.chunk_size,
                    ),
                    progress=progress,
                    cancelled=self.cancellations[job_id].is_set,
                )
                artifact = f"runs/{job_id}"
                if result["status"] == "cancelled":
                    self._update(job_id, status="cancelled", artifact=artifact)
                    return
            else:
                from .bridge import SmolOptions, create_smol_plan

                values = {"steps": req.steps, "batch_size": req.batch_size, "seed": req.seed}
                if req.python:
                    values["python"] = req.python
                create_smol_plan(
                    self._prepared(req.prepared_id),
                    self.output / "smol-plans" / job_id,
                    options=SmolOptions(**values),
                )
                artifact = f"smol-plans/{job_id}"
            self._update(job_id, status="completed", artifact=artifact)
        except Exception as exc:
            self._update(job_id, status="failed", error=str(exc))

    def _prepared(self, name):
        if not name:
            raise ValueError("Select a prepared bundle first")
        path = inside(self.output, f"prepared/{name}")
        if not (path / "manifest.json").is_file():
            raise ValueError("Prepared bundle is unavailable")
        return path

    def cancel(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            if not job or job["kind"] != "train" or job["status"] not in ("running", "queued"):
                raise ValueError("Only an active reference training job can be cancelled")
            self.cancellations[job_id].set()

    def snapshot(self):
        with self.lock:
            jobs = copy.deepcopy(list(self.jobs.values()))
        analysis_path = self.output / "analysis.json"
        analysis = load_json(analysis_path) if analysis_path.is_file() else None
        prepared = []
        for path in sorted((self.output / "prepared").glob("*/manifest.json")):
            manifest = load_json(path)
            if manifest.get("complete"):
                prepared.append(
                    {"id": path.parent.name, "split": load_json(path.parent / "split.json")}
                )
        runs = []
        for path in sorted((self.output / "runs").glob("*/run.json")):
            run = load_json(path)
            run["id"] = path.parent.name
            metric_path = path.parent / "metrics.jsonl"
            run["metrics"] = []
            if metric_path.exists():
                for line in metric_path.read_text(encoding="utf-8").splitlines():
                    try:
                        run["metrics"].append(json.loads(line))
                    except ValueError:
                        pass  # a writer may currently be appending the last line
            runs.append(run)
        return {
            "version": __version__,
            "dataset": self.root.name,
            "dataset_root": str(self.root),
            "workspace": str(self.output),
            "episodes": len(self.dataset.episodes),
            "frames": sum(x["length"] for x in self.dataset.episodes.values()),
            "cameras": visual_keys(self.dataset),
            "analysis": analysis,
            "prepared": prepared,
            "runs": runs,
            "jobs": jobs,
        }

    def close(self):
        for event in self.cancellations.values():
            event.set()
        self.pool.shutdown(wait=True, cancel_futures=True)


def make_server(workspace: Workspace, port=8765):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, data, mime="application/json", code=200):
            if not isinstance(data, bytes):
                data = json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.headers.get("Host") not in {
                f"127.0.0.1:{self.server.server_port}",
                f"localhost:{self.server.server_port}",
            }:
                return self.send({"error": "Local host required"}, code=403)
            try:
                url = urlparse(self.path)
                if url.path == "/api/state":
                    return self.send(workspace.snapshot())
                if url.path == "/api/image":
                    query = parse_qs(url.query)
                    ep, frame = int(query["episode"][0]), int(query["frame"][0])
                    key = query.get("camera", visual_keys(workspace.dataset))[0]
                    records = episode_records(workspace.dataset, ep)
                    if not 0 <= frame < len(records):
                        raise ValueError("Frame is out of range")
                    image = read_image(workspace.dataset, records[frame], key, size=256)
                    buffer = io.BytesIO()
                    image.save(buffer, format="PNG")
                    return self.send(buffer.getvalue(), "image/png")
                if url.path == "/api/predictions":
                    name = parse_qs(url.query)["run"][0]
                    path = inside(workspace.output / "runs", f"{name}/predictions.json")
                    return self.send(load_json(path))
                if url.path == "/":
                    content = (
                        files("vla_preflight")
                        .joinpath("web/index.html")
                        .read_text(encoding="utf-8")
                    )
                    content = content.replace("__SESSION_TOKEN__", workspace.token)
                    return self.send(content.encode("utf-8"), "text/html; charset=utf-8")
                assets = {
                    "/app.js": "text/javascript; charset=utf-8",
                    "/style.css": "text/css; charset=utf-8",
                }
                if url.path in assets:
                    return self.send(
                        files("vla_preflight").joinpath("web" + url.path).read_bytes(),
                        assets[url.path],
                    )
                return self.send({"error": "Not found"}, code=404)
            except (ValueError, KeyError, IndexError, OSError) as exc:
                self.send({"error": str(exc)}, code=400)

        def do_POST(self):
            host = self.headers.get("Host", "")
            allowed = {
                f"127.0.0.1:{self.server.server_port}",
                f"localhost:{self.server.server_port}",
            }
            if host not in allowed or self.headers.get("Origin") != f"http://{host}":
                return self.send({"error": "Local same-origin request required"}, code=403)
            if self.headers.get("X-Preflight-Token") != workspace.token:
                return self.send({"error": "Invalid session token"}, code=403)
            try:
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= 65536:
                    raise ValueError("Request body must be 1–65536 bytes")
                data = json.loads(self.rfile.read(size))
                if self.path == "/api/jobs":
                    request = JobRequest.model_validate(data)
                    return self.send({"id": workspace.start(request)}, code=202)
                if self.path == "/api/cancel":
                    workspace.cancel(data["id"])
                    return self.send({"cancel_requested": True})
                return self.send({"error": "Not found"}, code=404)
            except (ValueError, KeyError, TypeError, OSError) as exc:
                self.send({"error": str(exc)}, code=400)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve(root, output, *, port=8765, open_browser=False):
    if not 0 <= port <= 65535:
        raise ValueError("port must be 0–65535")
    workspace = Workspace(root, output)
    server = make_server(workspace, port)
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"VLA Workbench: {url}\nPress Ctrl+C to stop. Source dataset is read-only.", flush=True)
    if not (workspace.output / "analysis.json").exists():
        workspace.start(JobRequest(kind="analyze"))
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        workspace.close()
