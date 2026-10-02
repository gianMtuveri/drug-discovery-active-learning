import numpy as np
import pandas as pd
import pytest

from scripts.preprocessing.clean_bindingdb_all import (
    _assert_safe_output,
    aggregate_and_select_measurements,
    extract_long_measurements,
)
from scripts.preprocessing.prepare_target import prepare_target


def _source_row(target, smiles, *, ki=np.nan, kd=np.nan, ic50=np.nan):
    return {
        "Target Name": target,
        "Ligand SMILES": smiles,
        "Ki (nM)": ki,
        "Kd (nM)": kd,
        "IC50 (nM)": ic50,
    }


def _curate(rows):
    long, statistics = extract_long_measurements(pd.DataFrame(rows))
    selected, per_type = aggregate_and_select_measurements(long)
    return selected, per_type, statistics


def test_extracts_multiple_measurement_types_from_one_source_row():
    selected, per_type, statistics = _curate(
        [_source_row("Target A", "CCO", ki="5", kd="10", ic50="<20")]
    )
    assert set(per_type.affinity_type) == {"Ki", "Kd", "IC50"}
    assert statistics["valid_measurements"] == 3
    assert selected.loc[0, "affinity_type"] == "Ki"
    assert selected.loc[0, "available_affinity_types"] == "Ki|Kd|IC50"
    assert bool(selected.loc[0, "cross_type_resolved"])


def test_multiple_rows_aggregate_same_type_median_before_priority():
    selected, per_type, _ = _curate(
        [
            _source_row("Target A", "CCO", ki=1, ic50=100),
            _source_row("Target A", "OCC", ki=9, kd=50),
            _source_row("Target A", "CCO", ki=5),
        ]
    )
    ki = per_type[per_type.affinity_type == "Ki"].iloc[0]
    assert ki.aff_nM_median == 5
    assert ki.n_meas == 3
    assert selected.loc[0, "aff_nM_median"] == 5
    assert selected.loc[0, "n_meas"] == 3


@pytest.mark.parametrize(
    ("rows", "expected_type", "expected_value"),
    [
        (
            [_source_row("T", "CCO", ki=3, kd=2, ic50=1)],
            "Ki",
            3,
        ),
        (
            [_source_row("T", "CCO", kd=2, ic50=1)],
            "Kd",
            2,
        ),
        (
            [_source_row("T", "CCO", ic50=1)],
            "IC50",
            1,
        ),
    ],
)
def test_measurement_priority_and_fallback(rows, expected_type, expected_value):
    selected, _, _ = _curate(rows)
    assert selected.loc[0, "affinity_type"] == expected_type
    assert selected.loc[0, "aff_nM_median"] == expected_value


def test_missing_invalid_and_nonpositive_measurements_are_excluded():
    long, statistics = extract_long_measurements(
        pd.DataFrame(
            [
                _source_row("T", "CCO", ki="not measured", kd=0, ic50=-1),
                _source_row("T", "CCN", ki=">100"),
                _source_row("T", "invalid", ki=5),
                _source_row(None, "CCC", ki=5),
            ]
        )
    )
    assert len(long) == 1
    assert long.iloc[0].aff_nM == 100
    assert statistics["rows_without_valid_measurement"] == 1
    assert statistics["invalid_structure_rows"] == 1
    assert statistics["missing_target_or_smiles_rows"] == 1


def test_target_identity_separates_identical_structures():
    selected, _, _ = _curate(
        [
            _source_row("Target A", "CCO", ki=1),
            _source_row("Target B", "OCC", kd=2),
        ]
    )
    assert len(selected) == 2
    assert not selected.duplicated(["Target Name", "canonical_smiles"]).any()
    assert dict(zip(selected["Target Name"], selected.affinity_type)) == {
        "Target A": "Ki",
        "Target B": "Kd",
    }


def test_selection_is_deterministic_and_duplicate_free():
    rows = [
        _source_row("T", "CCO", ki=1, kd=4),
        _source_row("T", "OCC", ki=3, ic50=8),
        _source_row("T", "CCN", ic50=9),
    ]
    first, _, _ = _curate(rows)
    second, _, _ = _curate(rows)
    pd.testing.assert_frame_equal(first, second)
    assert not first.duplicated(["Target Name", "canonical_smiles"]).any()


def test_historical_master_and_target_paths_are_refused(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    historical_master = tmp_path / "data/processed/bindingdb_clean_all.parquet"
    historical_master.parent.mkdir(parents=True)
    historical_master.write_bytes(b"historical")
    with pytest.raises(ValueError, match="historical v0.5.0"):
        _assert_safe_output(historical_master, overwrite=True)
    assert historical_master.read_bytes() == b"historical"

    registry = tmp_path / "targets.yaml"
    registry.write_text("TEST:\n  query: Target A\n  match: exact\n")
    with pytest.raises(ValueError, match="historical v0.5.0"):
        prepare_target(
            "TEST",
            tmp_path / "unused.parquet",
            registry,
            tmp_path / "data/processed/targets",
        )


def test_corrected_target_artifacts_are_molecularly_aligned(tmp_path):
    master = tmp_path / "bindingdb_clean_all.parquet"
    registry = tmp_path / "targets.yaml"
    output = tmp_path / "v051/targets"
    selected, _, _ = _curate(
        [
            _source_row("Target A", "CCO", ki=10, ic50=100),
            _source_row("Target A", "CCN", kd=20),
        ]
    )
    selected.to_parquet(master, index=False)
    registry.write_text(
        "TEST:\n  query: Target A\n  match: exact\n  activity_threshold_nM: 500\n"
    )
    prepare_target("TEST", master, registry, output)

    target_dir = output / "TEST"
    clean = pd.read_parquet(target_dir / "clean.parquet")
    descriptors = pd.read_parquet(target_dir / "descriptors.parquet")
    with np.load(target_dir / "X_morgan.npz") as stored:
        fingerprint_smiles = stored["canonical_smiles"].astype(str)
        assert stored["X"].shape == (2, 2048)
    affinity = np.load(target_dir / "affinity.npy")
    assert np.array_equal(clean.canonical_smiles.to_numpy(), fingerprint_smiles)
    assert np.array_equal(clean.canonical_smiles.to_numpy(), descriptors.canonical_smiles.to_numpy())
    assert np.allclose(affinity, 9 - np.log10(clean.aff_nM_median.to_numpy()))
    assert np.array_equal(clean.label.to_numpy(), np.load(target_dir / "y_classification.npy"))
    assert np.array_equal(clean.label.to_numpy(), descriptors.label.to_numpy())

    second_output = tmp_path / "second/v051/targets"
    prepare_target("TEST", master, registry, second_output)
    second_dir = second_output / "TEST"
    pd.testing.assert_frame_equal(clean, pd.read_parquet(second_dir / "clean.parquet"))
    pd.testing.assert_frame_equal(
        descriptors,
        pd.read_parquet(second_dir / "descriptors.parquet"),
    )
    with np.load(second_dir / "X_morgan.npz") as second:
        assert np.array_equal(fingerprint_smiles, second["canonical_smiles"].astype(str))
        with np.load(target_dir / "X_morgan.npz") as first:
            assert np.array_equal(first["X"], second["X"])
    assert np.array_equal(affinity, np.load(second_dir / "affinity.npy"))
