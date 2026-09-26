import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from vla_preflight.audit import audit
from vla_preflight.cli import main
from vla_preflight.contract import Contract, from_lerobot, load_contract
from vla_preflight.demo import FAULTS, create_demo, write_json
from vla_preflight.report import Report


@pytest.fixture(params=["v2.1", "v3.0"])
def clean(tmp_path, request):
    return create_demo(tmp_path / "clean", version=request.param)


def scan(root, **kwargs):
    return audit(
        root / "dataset",
        contract=load_contract(root / "contract.json"),
        windows=root / "windows.jsonl",
        **kwargs,
    )


def codes(report):
    return set(report.findings)


def mutate_rows(root, change):
    path = sorted((root / "dataset/data").rglob("*.parquet"))[0]
    table = pq.read_table(path)
    records = table.to_pylist()
    change(records)
    pq.write_table(pa.Table.from_pylist(records, schema=table.schema), path)


def snapshot(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }


def test_clean_and_read_only(clean):
    before = snapshot(clean)
    report = scan(clean)
    assert report.status == "passed", report.to_dict()
    assert report.coverage["rows_scanned"] == 12
    assert report.complete
    assert snapshot(clean) == before


@pytest.mark.parametrize("fault,expected", FAULTS.items())
@pytest.mark.parametrize("version", ["v2.1", "v3.0"])
def test_fault_matrix(tmp_path, fault, expected, version):
    root = create_demo(tmp_path / fault, version=version, fault=fault)
    report = scan(root)
    assert expected in codes(report)
    assert report.status == "failed"
    assert report.findings[expected].examples


def test_partial_scope_is_explicit(clean):
    report = scan(clean, episodes={1})
    assert report.status == "passed"
    assert report.coverage["rows_scanned"] == 6
    assert report.coverage["scope"] == "selected episodes"


def test_shape_mismatch(clean):
    mutate_rows(clean, lambda r: r[2].update(action=[1, 2, 3]))
    assert "FEATURE_SHAPE" in codes(scan(clean))


def test_null_vector(clean):
    mutate_rows(clean, lambda r: r[2].update(action=None))
    assert "FEATURE_MISSING" in codes(scan(clean))


def test_frame_order(clean):
    mutate_rows(clean, lambda r: r[2].update(frame_index=1))
    assert "FRAME_SEQUENCE" in codes(scan(clean))


def test_timestamp_nan(clean):
    mutate_rows(clean, lambda r: r[2].update(timestamp=float("nan")))
    report = scan(clean)
    assert "TIMESTAMP_INVALID" in codes(report)
    json.dumps(report.to_dict(), allow_nan=False)


def test_reversed_time(clean):
    mutate_rows(clean, lambda r: r[2].update(timestamp=0.05))
    assert "TIMESTAMP_ORDER" in codes(scan(clean))


def test_stats_invalid_json(clean):
    (clean / "dataset/meta/stats.json").write_text("bad", encoding="utf-8")
    assert "STATS_INVALID" in codes(scan(clean))


def test_stats_missing_normalization(clean):
    path = clean / "dataset/meta/stats.json"
    stats = json.loads(path.read_text())
    del stats["action"]["std"]
    write_json(path, stats)
    assert "STATS_REQUIRED" in codes(scan(clean))


def test_quantile_requirement(clean):
    contract = Contract(normalization={"action": "QUANTILES"})
    report = audit(clean / "dataset", contract=contract)
    assert report.findings["STATS_REQUIRED"].count == 2


def test_constant_is_warning_not_error(clean):
    for path in (clean / "dataset/data").rglob("*.parquet"):
        table = pq.read_table(path)
        records = table.to_pylist()
        for row in records:
            row["action"][0] = 1
        pq.write_table(pa.Table.from_pylist(records, schema=table.schema), path)
    report = scan(clean)
    assert report.status == "warning"
    assert report.findings["CONSTANT_ACTION"].severity == "warning"


def test_metadata_count(clean):
    path = clean / "dataset/meta/info.json"
    obj = json.loads(path.read_text())
    obj["total_frames"] = 999
    write_json(path, obj)
    assert "METADATA_COUNT" in codes(scan(clean))


def test_corrupt_parquet(clean):
    path = next((clean / "dataset/data").rglob("*.parquet"))
    path.write_bytes(b"broken parquet")
    report = scan(clean)
    assert not report.complete
    assert "PARQUET_READ" in codes(report)


def test_bad_metadata_no_traceback(clean):
    (clean / "dataset/meta/info.json").write_text("[]", encoding="utf-8")
    assert "DATASET_METADATA" in codes(scan(clean))


def test_unknown_format(clean):
    path = clean / "dataset/meta/info.json"
    obj = json.loads(path.read_text())
    obj["codebase_version"] = "v9.0"
    write_json(path, obj)
    assert not scan(clean).complete


def test_path_traversal(clean):
    path = clean / "dataset/meta/info.json"
    obj = json.loads(path.read_text())
    obj["data_path"] = "../../outside.parquet"
    write_json(path, obj)
    report = scan(clean)
    assert "DATASET_METADATA" in codes(report)
    assert "escapes root" in report.findings["DATASET_METADATA"].examples[0]["detail"]


def test_bad_sampler_boolean(clean):
    path = clean / "windows.jsonl"
    obj = json.loads(path.read_text())
    obj["padding_mask"] = [0, 0, 1, 1]
    path.write_text(json.dumps(obj), encoding="utf-8")
    assert "WINDOW_TRACE_INVALID" in codes(scan(clean))


def test_unmasked_padding(clean):
    path = clean / "windows.jsonl"
    obj = json.loads(path.read_text())
    obj["padding_mask"][-1] = False
    path.write_text(json.dumps(obj), encoding="utf-8")
    assert "WINDOW_TARGET" in codes(scan(clean))


def test_no_contract_no_claim(clean):
    report = audit(clean / "dataset")
    assert "not checked" in report.coverage["training_contract"]
    assert "not checked" in report.coverage["sampler"]


def test_reports_escape_untrusted_text(tmp_path):
    report = Report()
    report.add("X", "error", "<script>alert(1)</script>", "<img src=x>", bad="<iframe>")
    assert "<script>" not in report.html()
    assert "&lt;script&gt;" in report.html()
    for _ in range(100):
        report.add("X", "error", "x", "y", row=1)
    assert len(report.findings["X"].examples) == 5
    assert report.findings["X"].count == 101


def test_outputs_cannot_touch_dataset(clean):
    before = snapshot(clean)
    assert (
        main(["audit", str(clean / "dataset"), "--output", str(clean / "dataset/meta/info.json")])
        == 2
    )
    assert snapshot(clean) == before


def test_cli_clean_json_and_html(clean, capsys):
    output, html = clean.parent / "report.json", clean.parent / "report.html"
    assert (
        main(
            [
                "audit",
                str(clean / "dataset"),
                "--contract",
                str(clean / "contract.json"),
                "--json",
                "--output",
                str(output),
                "--html",
                str(html),
            ]
        )
        == 0
    )
    obj = json.loads(capsys.readouterr().out)
    assert obj["status"] == "passed"
    assert json.loads(output.read_text())["status"] == "passed"
    assert "<html" in html.read_text()


def test_unknown_contract_keys():
    with pytest.raises(ValueError):
        Contract.model_validate({"action_dims": 2})


def test_contract_rejects_bool_and_duplicate_splits():
    with pytest.raises(ValueError):
        Contract(action_dim=True)
    with pytest.raises(ValueError):
        Contract(train_episodes=[0, 0])


def test_native_train_config(tmp_path):
    path = tmp_path / "train.json"
    write_json(
        path,
        {
            "policy": {
                "input_features": {
                    "observation.state": {"type": "STATE", "shape": [6]},
                    "observation.images.front": {"type": "VISUAL", "shape": [3, 224, 224]},
                },
                "output_features": {"action": {"type": "ACTION", "shape": [6]}},
                "normalization_mapping": {"ACTION": "MEAN_STD", "STATE": "MEAN_STD"},
            },
            "dataset": {"episodes": [0, 1]},
        },
    )
    contract = from_lerobot(path)
    assert contract.action_dim == 6
    assert contract.camera_keys == ["observation.images.front"]
    assert contract.normalization["action"] == "MEAN_STD"
    assert contract.train_episodes == [0, 1]


def test_unresolved_train_config(tmp_path):
    path = tmp_path / "train.json"
    write_json(path, {"policy": {"type": "smolvla"}})
    with pytest.raises(ValueError, match="Unresolved"):
        from_lerobot(path)


def test_demo_never_overwrites(tmp_path):
    create_demo(tmp_path / "demo")
    before = snapshot(tmp_path)
    assert main(["demo", str(tmp_path / "demo")]) == 2
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("tolerance", [-1, float("nan"), float("inf")])
def test_invalid_tolerance(clean, tolerance):
    with pytest.raises(ValueError):
        scan(clean, timestamp_tolerance=tolerance)


def test_global_index_mismatch(clean):
    mutate_rows(clean, lambda r: r[2].update(index=500))
    assert "GLOBAL_INDEX" in codes(scan(clean))


def test_nonstring_path_template(clean):
    path = clean / "dataset/meta/info.json"
    obj = json.loads(path.read_text())
    obj["data_path"] = 123
    write_json(path, obj)
    assert not scan(clean).complete


def test_nonfinite_evidence_is_json_safe():
    report = Report()
    report.add("BAD", "error", "bad", "fix", value=float("nan"))
    assert '"nan"' in json.dumps(report.to_dict(), allow_nan=False)


def test_native_malformed_dataset(tmp_path):
    path = tmp_path / "train.json"
    write_json(
        path,
        {
            "policy": {"output_features": {"action": {"type": "ACTION", "shape": [2]}}},
            "dataset": "wrong",
        },
    )
    with pytest.raises(ValueError, match="dataset must"):
        from_lerobot(path)


def test_strict_warning_exit(clean, capsys):
    stats_path = clean / "dataset/meta/stats.json"
    stats = json.loads(stats_path.read_text())
    stats["action"]["std"][0] = 0
    write_json(stats_path, stats)
    assert main(["audit", str(clean / "dataset")]) == 0
    assert main(["audit", str(clean / "dataset"), "--strict"]) == 1


def test_bad_json_contract_returns_two(clean):
    path = clean / "contract.json"
    path.write_text("bad json", encoding="utf-8")
    assert main(["audit", str(clean / "dataset"), "--contract", str(path)]) == 2


def test_report_cannot_overwrite_contract(clean):
    path = clean / "contract.json"
    original = path.read_bytes()
    assert (
        main(["audit", str(clean / "dataset"), "--contract", str(path), "--output", str(path)]) == 2
    )
    assert path.read_bytes() == original


def test_empty_windows_trace(clean):
    (clean / "windows.jsonl").write_text("", encoding="utf-8")
    assert "WINDOW_TRACE_EMPTY" in codes(scan(clean))


def test_metadata_motor_names(clean):
    path = clean / "dataset/meta/info.json"
    obj = json.loads(path.read_text())
    obj["features"]["action"]["names"] = {"motors": ["joint_a", "joint_b"]}
    write_json(path, obj)
    assert scan(clean).status == "passed"


def test_v3_multiple_data_shards(tmp_path):
    root = create_demo(tmp_path / "shared")
    path = root / "dataset/data/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    # Episode 0 stays in shard 0; episode 1 occupies shard 1.
    pq.write_table(table.slice(0, 6), path)
    pq.write_table(table.slice(6), path.with_name("file-001.parquet"))
    meta = root / "dataset/meta/episodes/chunk-000/file-000.parquet"
    records = pq.read_table(meta).to_pylist()
    records[1]["data/file_index"] = 1
    pq.write_table(pa.Table.from_pylist(records), meta)
    assert scan(root).status == "passed"


def test_unsupported_normalization_feature():
    with pytest.raises(ValueError):
        Contract(normalization={"next.reward": "MEAN_STD"})


def test_numeric_strings_are_not_accepted(clean):
    path = sorted((clean / "dataset/data").rglob("*.parquet"))[0]
    rows = pq.read_table(path).to_pylist()
    for row in rows:
        row["action"] = [str(x) for x in row["action"]]
    pq.write_table(pa.Table.from_pylist(rows), path)
    assert "FEATURE_TYPE" in scan(clean).findings
