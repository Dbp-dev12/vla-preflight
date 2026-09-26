"""Image access for the workbench and tiny reference trainer; imports are lazy."""

from __future__ import annotations

import io

from .dataset import inside


def visual_keys(ds):
    return [k for k, v in ds.features.items() if v.get("dtype") in ("image", "video")]


def read_image(ds, row, key, *, size=32):
    try:
        from PIL import Image
    except ImportError as exc:
        raise ValueError('Image support requires: pip install -e ".[train]"') from exc
    if key not in visual_keys(ds):
        raise ValueError("Select a declared image/video feature")
    if ds.features[key]["dtype"] == "image":
        value = row.get(key)
        if not isinstance(value, dict):
            raise ValueError("Image feature must contain bytes or a dataset-relative path")
        if value.get("bytes") is not None:
            source = io.BytesIO(value["bytes"])
        elif value.get("path"):
            source = inside(ds.root, value["path"])
        else:
            raise ValueError("Missing image bytes/path")
        with Image.open(source) as img:
            return img.convert("RGB").resize((size, size))
    try:
        import av
    except ImportError as exc:
        raise ValueError('Video decoding requires: pip install -e ".[train]"') from exc
    ep = row["episode_index"]
    reference = next(x for x in ds.video_references({ep}) if x[1] == key)
    timestamp = float(row["timestamp"])
    if ds.version == "v3.0":
        timestamp += ds.episodes[ep][f"videos/{key}/from_timestamp"]
    with av.open(str(reference[2])) as container:
        stream = container.streams.video[0]
        container.seek(max(0, int(timestamp * av.time_base)), backward=True)
        for frame in container.decode(stream):
            if frame.time is not None and frame.time + 0.5 / ds.fps >= timestamp:
                return frame.to_image().convert("RGB").resize((size, size))
    raise ValueError("Could not decode an image at the requested timestamp")
