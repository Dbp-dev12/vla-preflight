# Contracts and action-chunk traces

## Audit contract

Pass `--contract examples/contract.json`. This JSON belongs to VLA Preflight, **not**
to a training framework. Unknown fields are rejected. Fields are optional, but
checks are only performed for expectations you actually supply.

- `schema_version`: 1.
- `action_dim`, `state_dim`: positive integer vector dimensions.
- `fps`: positive, finite expected data frequency.
- `camera_keys`: exact dataset visual keys, before a trainer-specific rename.
- `normalization`: dataset feature → `IDENTITY`, `MEAN_STD`, `MIN_MAX`, `QUANTILES`.
  The audit checks statistics for `action` and `observation.state` only.
- `train_episodes`, `validation_episodes`: optional nonempty, unique integer ID lists.
  If both are supplied, the tool checks overlap. It does not detect duplicate content
  stored under different IDs, nor infer a validation split from a training config.
- `training`, `deployment`: optional physical-semantic declarations. Corresponding
  fields are compared only when explicitly present. Supported fields: `action_names`,
  `action_units`, `action_mode` (`absolute`/`delta`/`velocity`), `coordinate_frame`,
  `gripper_convention`. Names/units are per action dimension. These are declarations,
  not measurements or inspected processor implementations.

The JSON Schema is in [contract.schema.json](../examples/contract.schema.json).

## Resolved LeRobot train config

`--train-config train_config.json` reads `policy.input_features`,
`policy.output_features`, `policy.normalization_mapping` and `dataset.episodes`.
It requires resolved feature shapes. An unresolved config is rejected rather than
silently treated as compatible. Only standard action/state and visual feature fields
are extracted; extra trainer fields are not executed or audited.

The config must describe the same unremapped dataset features. Custom processors,
camera renames, relative-action transformations and checkpoint-embedded statistics
are **not** reproduced. Write a separate audit contract when using those features.
This import does not check whether the config's dataset repo ID matches the local
directory; you select the directory explicitly.

## Actual sampler trace (JSONL)

Export targets from your **actual training dataloader**, then supply `--windows trace.jsonl`.
Without this trace no action-chunk check is performed. Merely generating a correct
trace independently from your sampler cannot establish that the sampler is correct.

One line, for an episode of length 6:

```json
{"episode_index":0,"anchor_frame":4,"target_episode_indices":[0,0,0,0],"target_frame_indices":[4,5,5,5],"padding_mask":[false,false,true,true]}
```

The v1 trace convention is intentionally narrow:

1. The first action target is at `anchor_frame` (offset 0).
2. Subsequent targets are contiguous integer frames, one frame apart.
3. Beyond the episode end, repeat the last frame and set `padding_mask=true`.
4. Every target retains the original episode ID, including masked targets.
5. The boolean is a *padding mask*, not a training loss mask (inverse convention).

If your sampler starts at t+1, strides, resamples, uses zero padding, or uses a different
mask convention, translate its trace explicitly or do not use this check. The tool
does not claim that those alternative designs are inherently wrong.

Example instrumentation, placed where actual target row IDs and masks are available:

```python
import json

# Values must come from the sampler, not be recomputed with Preflight's rules.
record = {
    "episode_index": int(observation_episode_id),
    "anchor_frame": int(observation_frame_id),
    "target_episode_indices": [int(x) for x in sampled_episode_ids],
    "target_frame_indices": [int(x) for x in sampled_frame_ids],
    "padding_mask": [bool(x) for x in actual_padding_mask],
}
trace_file.write(json.dumps(record) + "\n")
```

## Before-training integration

No framework migration is needed. From Python:

```python
from pathlib import Path
from vla_preflight.audit import audit
from vla_preflight.contract import load_contract

result = audit(Path("my-dataset"), contract=load_contract(Path("contract.json")))
result.write(Path("reports/preflight.json"), dataset_root=Path("my-dataset"))
if not result.complete or result.status != "passed":
    raise SystemExit("Inspect preflight findings before launching training")
# Launch your existing training process here.
```

This gate intentionally blocks warnings too; users can change that policy after
reviewing known benign warnings. It does not make a training-success guarantee.

