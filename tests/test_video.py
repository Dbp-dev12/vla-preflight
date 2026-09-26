import json
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from vla_preflight.audit import audit
from vla_preflight.demo import create_demo, write_json


@pytest.fixture(params=["v2.1", "v3.0"])
def video_dataset(tmp_path, request):
    root = create_demo(tmp_path / "demo", version=request.param) / "dataset"
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    key = "observation.images.front"
    info["features"][key] = {"dtype": "video", "shape": [32, 32, 3]}
    if request.param == "v2.1":
        info["video_path"] = (
            "videos/{video_key}/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.mp4"
        )
        videos = [root / f"videos/{key}/chunk-000/episode_{ep:06d}.mp4" for ep in range(2)]
    else:
        info["video_path"] = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
        path = root / "meta/episodes/chunk-000/file-000.parquet"
        rows = pq.read_table(path).to_pylist()
        for ep in rows:
            ep.update(
                {
                    f"videos/{key}/chunk_index": 0,
                    f"videos/{key}/file_index": 0,
                    f"videos/{key}/from_timestamp": ep["episode_index"] * 0.6,
                    f"videos/{key}/to_timestamp": (ep["episode_index"] + 1) * 0.6,
                }
            )
        pq.write_table(pa.Table.from_pylist(rows), path)
        videos = [root / f"videos/{key}/chunk-000/file-000.mp4"]
    write_json(info_path, info)
    return root, videos


def test_missing_video(video_dataset):
    root, _ = video_dataset
    assert "VIDEO_MISSING" in audit(root).findings
    report = audit(root, video="skip")
    assert "VIDEO_MISSING" not in report.findings
    assert report.coverage["video_check"] == "skip"


def test_absent_ffprobe_marks_incomplete(video_dataset, monkeypatch):
    root, _ = video_dataset
    monkeypatch.setattr("vla_preflight.audit.shutil.which", lambda _: None)
    report = audit(root, video="probe")
    assert not report.complete
    assert "FFPROBE_UNAVAILABLE" in report.findings


def test_probe_contract_and_cache(video_dataset, monkeypatch):
    root, files = video_dataset
    for path in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not a real video; subprocess is stubbed in this unit test")
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        assert kwargs["timeout"] == 30
        assert "shell" not in kwargs
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "streams": [{"codec_type": "video", "width": 32, "height": 32}],
                    "format": {"duration": "2.0"},
                }
            ),
        )

    monkeypatch.setattr("vla_preflight.audit.shutil.which", lambda _: "ffprobe")
    monkeypatch.setattr("vla_preflight.audit.subprocess.run", run)
    report = audit(root, video="probe")
    assert report.status == "passed"
    assert len(calls) == len(files)
