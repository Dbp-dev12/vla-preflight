from types import SimpleNamespace

import numpy as np
import pytest

from vla_preflight.media import read_image


def test_real_video_decode_respects_episode_offset(tmp_path):
    av = pytest.importorskip("av")
    pytest.importorskip("PIL")
    path = tmp_path / "clip.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        for i in range(10):
            frame = av.VideoFrame.from_ndarray(
                np.full((32, 32, 3), i * 20, dtype=np.uint8), format="rgb24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    ds = SimpleNamespace(
        root=tmp_path,
        version="v3.0",
        fps=10,
        features={"camera": {"dtype": "video"}},
        episodes={0: {"videos/camera/from_timestamp": 0.3}},
        video_references=lambda _: iter([(0, "camera", path, 1.0)]),
    )
    # Relative .2 + episode start .3 must select the .5s frame, not .2s.
    result = np.asarray(read_image(ds, {"episode_index": 0, "timestamp": 0.2}, "camera"))
    assert abs(float(result.mean()) - 100) < 5


def test_path_based_image_cannot_escape_dataset(tmp_path):
    pytest.importorskip("PIL")
    ds = SimpleNamespace(root=tmp_path, features={"camera": {"dtype": "image"}})
    with pytest.raises(ValueError, match="escapes root"):
        read_image(ds, {"camera": {"path": "../outside.png"}}, "camera")
