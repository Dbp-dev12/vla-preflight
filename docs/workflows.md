# Workflow and artifact contracts

## Preparation

`prepare` requires a completed audit without errors; warnings remain in preflight.json. Exact numeric action/state sequences plus episode task labels form duplicate groups. Whole groups are randomly split with a fixed seed. Validation fraction is approximate by group count, not frames; at least two distinct groups are required. No image deduplication or task stratification is provided.

Normalization fits training episodes only: population std with epsilon floor 1e-6. Mean/std/min/max use all valid rows. Quantiles use a seeded reservoir of at most 8,192 vectors and are labeled exact or estimated.

SHA-256 covers metadata, referenced Parquet, path-based images and optional videos. Embedded image bytes are inside Parquet hashes. `--video skip` omits video hashing. Source fingerprints and prepared artifacts are checked before training. This is local consistency checking, not signed provenance; concurrent source modification while running is unsupported.

## Reference training

The random-initialized model concatenates CNN features, averaged byte embeddings and state features. One RGB camera is resized to 32×32. Instructions truncate at 64 UTF-8 bytes, potentially splitting multibyte characters; no pretrained language understanding is provided. AdamW, gradient clipping and nonfinite checks are used.

Samples come only from training episodes. Action targets clamp to the same episode's last frame; padded offsets have zero loss weight. Validation predicts the first action of each chunk and reports MAE/RMSE in original units, plus mean-action and zero-action baselines.

| Artifact | Meaning |
|---|---|
| run.json | Configuration, identity, status, initial/final/best validation, diagnostics |
| metrics.jsonl | Training loss and validation MAE at logged steps |
| checkpoint.pt | Latest model, optimizer, step, RNG; atomic replacement |
| best.pt | Best validation candidate in this run, including initialization |
| sampler-trace.jsonl | First actual batch's targets and masks, not all training samples |
| predictions.json | At most 200 evenly sampled held-out predictions/targets |

Only load checkpoints you trust. Resume requires the same bundle/camera/model shape and a larger total step target. CPU tests verify exact resumed versus uninterrupted updates with unchanged settings. Altering batch size or learning rate changes the experiment. Best-model selection restarts at the resumed checkpoint; it does not import the previous best.pt. CUDA reproducibility and memory use are unvalidated. Cancel stops between updates and saves state.

## Local workbench

`studio DATASET --workspace OUTPUT --port 8765 --open` starts a loopback server and one background job at a time. The UI is Chinese. Use a distinct workspace for each dataset. Restarted job history marks previously running jobs interrupted. Close with Ctrl+C; training is asked to cancel.

## Compute and physical evaluation

`doctor` records a shareable compute inventory without machine identity or absolute paths. Its VRAM tier is guidance only; it is not a pretrained-model memory benchmark.

`robot-plan` validates a strict robot declaration, hashes available calibration files and blocks the plan unless the emergency stop, cleared workspace, low-speed first run and human supervision are all declared. It generates argument arrays and a runbook but never imports a hardware driver or executes a command.

`rollout-eval` validates unique episode IDs and reports observed success, Wilson 95% intervals, duration, interventions and failure modes overall and by task/checkpoint. These are task-level observations, not a causal comparison or safety certificate. See [hardware.md](hardware.md).

Host checks, same-origin checks and session tokens protect mutation endpoints. Do not expose this single-user server to a network. Paths, task text and evidence remain local; exported reports should be reviewed before sharing.

## Export and experimental SmolVLA

Export selects complete episodes, materializes path-based images, copies videos unchanged, reindexes IDs, preserves timestamps and recomputes numeric statistics. export-status.json is complete only after local verification. Visual statistics are not produced. Shared video files can contain excluded footage; this is not privacy redaction.

`smol-plan` exports only training episodes and constructs an external LeRobot command: frozen vision encoder, expert-only training, gradient checkpointing, batch size 1 and eight-step gradient accumulation by default, uploads/W&B disabled. `smol-launch` explicitly executes it and records external.log/status; it may download weights. A successful process must also leave a safetensors checkpoint.

The bridge has not been validated against a live pretrained-model run or upstream dataset loader. No external-model held-out evaluation, camera remapping, arbitrary processors, or guaranteed memory configuration is provided. Review upstream requirements before execution. Tiny VLA is the fully exercised training backend.

## Resources

Audit streams numeric Parquet. Profiling retains numeric rows in memory; avoid unbounded corpora. Training preloads selected-camera 32×32 uint8 images and tensors, at most 50,000 frames by default. Per-frame video decoding can be slow. No distributed loader or resumable export is implemented; use manageable subsets.
