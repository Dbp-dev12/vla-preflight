import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("PIL")

from vla_preflight.analysis import prepare  # noqa: E402
from vla_preflight.learning_demo import learning_demo  # noqa: E402
from vla_preflight.training import (  # noqa: E402
    TrainOptions,
    batch,
    compare_runs,
    load_training_data,
    train,
)
from vla_preflight.workflow_io import load_prepared  # noqa: E402


@pytest.fixture
def prepared(tmp_path):
    root = learning_demo(tmp_path / "source", episodes=8, frames=8)
    path = tmp_path / "prepared"
    prepare(root, path)
    return path


def test_chunk_padding_never_crosses_episode(prepared):
    ds, _, _, normal = load_prepared(prepared)
    data = load_training_data(ds, normal, None, 1000)
    index = data["lookup"][0, 7]
    *_, target, mask, traces = batch(data, np.array([index]), ds, 4, "cpu")
    assert target.shape == (1, 4, 2)
    assert mask.tolist() == [[False, True, True, True]]
    assert traces[0]["target_episode_indices"] == [0] * 4
    assert traces[0]["target_frame_indices"] == [7] * 4


def test_resume_matches_uninterrupted_updates(prepared, tmp_path):
    options = dict(batch_size=4, log_every=2, seed=9)
    full = train(prepared, tmp_path / "full", options=TrainOptions(steps=6, **options))
    train(prepared, tmp_path / "first", options=TrainOptions(steps=3, **options))
    resumed = train(
        prepared,
        tmp_path / "resumed",
        options=TrainOptions(steps=6, **options),
        resume=tmp_path / "first/checkpoint.pt",
    )
    left = torch.load(tmp_path / "full/checkpoint.pt", weights_only=True)
    right = torch.load(tmp_path / "resumed/checkpoint.pt", weights_only=True)
    for key in left["model"]:
        torch.testing.assert_close(left["model"][key], right["model"][key], rtol=0, atol=0)
    assert full["validation"] == resumed["validation"]
    assert full["normalization_roundtrip_max_error"] < 1e-12
    assert (tmp_path / "full/best.pt").is_file()
    assert full["best_validation"]["mae"] <= full["validation"]["mae"]
    assert compare_runs([tmp_path / "full", tmp_path / "resumed"])["same_prepared_bundle"]


def test_cancel_saves_checkpoint_and_bundle_mismatch_rejected(prepared, tmp_path):
    result = train(prepared, tmp_path / "cancelled", cancelled=lambda: True)
    assert result["status"] == "cancelled" and result["completed_steps"] == 0
    assert (tmp_path / "cancelled/checkpoint.pt").is_file()
    with pytest.raises(ValueError, match="different prepared bundle or model shape"):
        train(
            prepared,
            tmp_path / "wrong",
            options=TrainOptions(chunk_size=2),
            resume=tmp_path / "cancelled/checkpoint.pt",
        )


def test_real_gradient_updates_change_weights(prepared, tmp_path):
    train(prepared, tmp_path / "initial", cancelled=lambda: True)
    result = train(prepared, tmp_path / "trained", options=TrainOptions(steps=4))
    before = torch.load(tmp_path / "initial/checkpoint.pt", weights_only=True)["model"]
    after = torch.load(tmp_path / "trained/checkpoint.pt", weights_only=True)["model"]
    for prefix in ("vision.", "language.", "state.", "head."):
        assert any(not torch.equal(before[k], after[k]) for k in before if k.startswith(prefix))
    assert result["status"] == "completed"
