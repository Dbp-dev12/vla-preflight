# Checks, evidence and limits

Errors are explicit schema or declared-contract violations. Warnings indicate
incomplete evidence or a potentially benign condition. Each code aggregates a count
and at most five evidence examples; counts are check occurrences, not distinct root causes.

| Codes | What is checked |
|---|---|
| `DATASET_METADATA`, `METADATA_COUNT` | Supported version/backend, metadata fields and counters |
| `DATA_FILE_MISSING`, `PARQUET_READ` | Metadata-referenced shards exist and numeric batches are readable |
| `EPISODE_ID`, `EPISODE_LENGTH` | Row IDs exist in metadata and scanned episode lengths match |
| `FRAME_SEQUENCE`, `GLOBAL_INDEX` | Consecutive frame IDs starting at zero; global offsets match |
| `TIMESTAMP_INVALID`, `TIMESTAMP_ORDER`, `TIMESTAMP_FPS` | Finite/increasing episode-relative time, consistent with declared fps |
| `FEATURE_MISSING`, `FEATURE_SHAPE`, `FEATURE_TYPE`, `NONFINITE` | Declared numeric features present, numeric, finite, correct shape |
| `CONSTANT_ACTION` | Exactly constant action dimension across the scanned scope (warning) |
| `CONFIG_DIMENSION`, `CONFIG_CAMERA`, `CONFIG_FPS` | Explicit expectations vs metadata |
| `SPLIT_OVERLAP`, `SPLIT_UNKNOWN` | Explicit split episode IDs overlap or do not exist |
| `SEMANTICS_MISMATCH`, `SEMANTICS_UNKNOWN` | Explicit training/deployment declarations differ or are one-sided |
| `ACTION_ORDER`, `ACTION_ORDER_UNKNOWN` | Declared training joint order vs dataset names (list or `motors` list) |
| `STATS_MISSING`, `STATS_REQUIRED`, `STATS_INVALID` | Stored aggregate statistics exist, have expected vector shapes and finite values |
| `STATS_NEGATIVE_STD`, `STATS_ZERO_STD`, `STATS_RANGE` | Standard deviations and min/max or quantile ordering |
| `NORMALIZATION_FEATURE` | Contract refers to an unknown dataset feature |
| `WINDOW_CROSS_EPISODE`, `WINDOW_TARGET` | Exported sampler targets respect the documented convention |
| `WINDOW_TRACE_INVALID`, `WINDOW_TRACE_EMPTY`, `WINDOW_TRACE_READ` | Actual trace supplied and readable with correct types |
| `VIDEO_MISSING`, `VIDEO_METADATA` | Referenced video files and v3 segment metadata |
| `FFPROBE_UNAVAILABLE`, `VIDEO_PROBE` | Requested optional container probing was possible and succeeded |
| `VIDEO_DURATION_UNKNOWN`, `VIDEO_SEGMENT_RANGE` | v3 segment end fits probed duration with a one-frame tolerance |

## Supported layouts

- v2.1: `meta/info.json`, `meta/episodes.jsonl`, metadata-templated per-episode Parquet.
- v3.0: `meta/info.json`, `meta/episodes/**/*.parquet`, shared Parquet shards resolved
  via `data/chunk_index` and `data/file_index`. Complete metadata is required even when
  selecting a subset of episodes. Canonical global index ranges must be contiguous.
- Optional aggregate `meta/stats.json`. Some v2.1 datasets only store per-episode
  statistics; v0.1 does **not** aggregate `episodes_stats.jsonl`, and reports the missing
  aggregate. Without a normalization requirement this is a warning; with one it is an error.
- Numeric action/state vectors only. Image feature payloads, labels and task semantics are not audited.

## Scope is part of the result

`scan_complete=true` means the requested scan completed, **not** that every possible
check or every episode was checked. Read `coverage.scope`, selected episode IDs,
video mode, contract and sampler coverage. Selecting episodes still requires valid
complete metadata. Unselected rows in a shared shard are ignored after reading their IDs.

The tool scans referenced files; it does not detect orphan shards absent from metadata.
It retains metadata/counters per episode, but does not materialize the full dataset or
decode image columns. Speed and peak memory are not benchmarked for large-scale datasets.

Statistics are validated structurally, not recomputed. They may originate from another
dataset or checkpoint; this release cannot establish that provenance. Only action/state
normalization statistics are checked, even if other numeric features are present.

Timestamps default to a 1 ms tolerance against `frame_index/fps`. A dataset with a
different legitimate clock policy may need a documented tolerance adjustment. This
does not verify the timing of the visual content relative to the action.

Video mode `exists` validates paths, not codecs. Mode `probe` checks for a video stream
using ffprobe and optionally v3 segment duration bounds. It does not decode every frame,
verify camera calibration, or measure semantic synchronization. An unavailable ffprobe
marks a requested probe scan incomplete and exits 2.

## References inspected for compatibility

- [LeRobot dataset implementation](https://github.com/huggingface/lerobot/blob/main/src/lerobot/datasets/lerobot_dataset.py)
- [LeRobot layout utilities](https://github.com/huggingface/lerobot/blob/main/src/lerobot/datasets/utils.py)
- [LeRobot training config](https://github.com/huggingface/lerobot/blob/main/src/lerobot/configs/train.py)
- [OpenPI normalization documentation](https://github.com/Physical-Intelligence/openpi/blob/main/docs/norm_stats.md)

These links are upstream references, not endorsements. The public smoke test uses a pinned
dataset revision; future format changes may require new adapters.

