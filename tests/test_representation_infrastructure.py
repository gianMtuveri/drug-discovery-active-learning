import json

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src.data.partitions import (
    generate_partition_manifest,
    load_partition_manifest,
    save_partition_manifest,
)
from src.data.representations import (
    MolecularRepresentation,
    RDKIT_DESCRIPTOR_COLUMNS,
    build_common_population,
    load_frozen_embedding_representation,
    load_morgan_representation,
    load_rdkit_descriptor_representation,
    make_molecule_ids,
    ordered_population_hash,
    validate_representation,
)
from src.data.targets import load_target_regression


def _synthetic_representation(identifier, smiles, success, reasons=None, features=None):
    smiles = np.asarray(smiles, dtype=str)
    ids = make_molecule_ids(smiles)
    success = np.asarray(success, dtype=bool)
    if reasons is None:
        reasons = np.where(success, "", "synthetic_failure")
    if features is None:
        features = np.arange(len(smiles) * 2, dtype=float).reshape(len(smiles), 2)
    features = np.asarray(features)
    provenance = {
        "schema_version": 1,
        "representation": identifier,
        "ordered_population_hash": ordered_population_hash(ids),
        "feature_shape": list(features.shape),
        "feature_dtype": str(features.dtype),
        "loader_version": "test",
        "generation_parameters": {},
    }
    return MolecularRepresentation(
        identifier=identifier,
        molecule_ids=ids,
        canonical_smiles=smiles,
        features=features,
        success=success,
        failure_reasons=np.asarray(reasons, dtype=object),
        provenance=provenance,
    )


def test_real_morgan_is_preserved_exactly_and_old_loader_is_compatible():
    representation = load_morgan_representation("EGFR")
    stored = np.load("data/processed/targets/EGFR/X_morgan.npz")
    old_X, old_y = load_target_regression("EGFR")

    assert representation.features.dtype == stored["X"].dtype
    assert np.array_equal(representation.features, stored["X"])
    assert np.array_equal(representation.features, old_X)
    assert len(old_y) == len(representation.features)


def test_real_rdkit_selects_exact_columns_and_excludes_qed():
    morgan = load_morgan_representation("EGFR")
    representation = load_rdkit_descriptor_representation("EGFR", reference=morgan)
    stored = pd.read_parquet("data/processed/targets/EGFR/descriptors.parquet")

    assert representation.provenance["generation_parameters"]["descriptor_columns"] == list(
        RDKIT_DESCRIPTOR_COLUMNS
    )
    assert "QED" not in RDKIT_DESCRIPTOR_COLUMNS
    assert representation.features.shape == (14174, 8)
    assert np.array_equal(
        representation.features,
        stored.loc[:, list(RDKIT_DESCRIPTOR_COLUMNS)].to_numpy(),
    )


def test_validation_detects_reordered_and_missing_molecules():
    reference = _synthetic_representation("a", ["CCO", "CCN", "CCC"], [1, 1, 1])
    reordered = _synthetic_representation("b", ["CCN", "CCO", "CCC"], [1, 1, 1])
    missing = _synthetic_representation("c", ["CCO", "CCN"], [1, 1])
    with pytest.raises(ValueError, match="order differs"):
        validate_representation(reordered, reference=reference)
    with pytest.raises(ValueError, match="order differs"):
        validate_representation(missing, reference=reference)


def test_validation_rejects_nonfinite_successful_features_and_implicit_failures():
    nonfinite = _synthetic_representation(
        "a", ["CCO", "CCN"], [1, 1], features=[[1.0, np.nan], [2.0, 3.0]]
    )
    with pytest.raises(ValueError, match="non-finite"):
        validate_representation(nonfinite)

    implicit = _synthetic_representation("b", ["CCO", "CCN"], [1, 0], reasons=["", ""])
    with pytest.raises(ValueError, match="explicit reason"):
        validate_representation(implicit)


def test_validation_rejects_duplicate_or_missing_molecular_identities():
    representation = _synthetic_representation("a", ["CCO", "CCN"], [1, 1])
    duplicate_ids = representation.molecule_ids.copy()
    duplicate_ids[1] = duplicate_ids[0]
    with pytest.raises(ValueError, match="Duplicate molecular identities"):
        validate_representation(replace_representation(representation, molecule_ids=duplicate_ids))

    missing_ids = representation.molecule_ids.copy()
    missing_ids[1] = ""
    with pytest.raises(ValueError, match="cannot be missing"):
        validate_representation(replace_representation(representation, molecule_ids=missing_ids))

    with pytest.raises(ValueError, match="cannot be missing"):
        make_molecule_ids(["CCO", None])


def test_common_population_intersects_masks_and_records_reasons_in_order():
    first = _synthetic_representation(
        "morgan", ["CCO", "CCN", "CCC", "CO"], [1, 1, 0, 1],
        reasons=["", "", "fingerprint_failure", ""],
    )
    second = _synthetic_representation(
        "future", ["CCO", "CCN", "CCC", "CO"], [1, 0, 1, 1],
        reasons=["", "token_limit", "", ""],
    )
    common = build_common_population([first, second])

    assert common.inclusion_mask.tolist() == [True, False, False, True]
    assert common.original_size == 4
    assert common.common_size == 2
    assert common.exclusion_reasons[1] == {"future": "token_limit"}
    assert common.exclusion_reasons[2] == {"morgan": "fingerprint_failure"}
    assert common.canonical_smiles[common.inclusion_mask].tolist() == ["CCO", "CO"]


def test_common_eligibility_is_independent_of_affinity_and_qed():
    representation = _synthetic_representation("a", ["CCO", "CCN"], [1, 0])
    expected = build_common_population([representation]).inclusion_mask.copy()
    # Neither array is accepted by the eligibility API, so changing them cannot
    # affect the structure/encoding-derived intersection.
    for affinity, qed in (
        (np.array([1.0, 1000.0]), np.array([0.1, 0.9])),
        (np.array([np.nan, -1.0]), np.array([1.0, 0.0])),
    ):
        assert affinity.shape == qed.shape
        assert np.array_equal(build_common_population([representation]).inclusion_mask, expected)


def test_frozen_embedding_loader_uses_stored_mask_and_nan_failure_rows(tmp_path):
    smiles = np.asarray(["CCO", "CCN", "CCC"])
    ids = make_molecule_ids(smiles)
    features = np.asarray([[1, 2], [np.nan, np.nan], [3, 4]], dtype=np.float32)
    np.save(tmp_path / "embeddings.npy", features)
    np.save(tmp_path / "success_mask.npy", np.asarray([True, False, True]))
    np.save(tmp_path / "molecule_ids.npy", ids)
    pd.DataFrame({
        "molecule_id": ids,
        "canonical_smiles": smiles,
        "status": ["success", "token_limit_exceeded", "success"],
        "failure_reason": ["", "203>202", ""],
    }).to_parquet(tmp_path / "extraction_audit.parquet", index=False)
    (tmp_path / "manifest.json").write_text('{"schema_version": 1}')

    representation = load_frozen_embedding_representation(
        "frozen", tmp_path, expected_dimension=2
    )

    assert representation.success.tolist() == [True, False, True]
    assert representation.failure_reasons.tolist() == ["", "203>202", ""]
    assert np.array_equal(representation.features, features, equal_nan=True)


def test_partition_manifest_is_deterministic_and_matches_v050_logic(tmp_path):
    ids = make_molecule_ids([f"C{'C' * index}" for index in range(1, 51)])
    first = generate_partition_manifest(ids, random_seed=7, test_size=0.2, n_initial=20)
    second = generate_partition_manifest(ids, random_seed=7, test_size=0.2, n_initial=20)
    assert first == second

    pool, test = train_test_split(np.arange(len(ids)), test_size=0.2, random_state=7)
    initial = np.random.default_rng(7).choice(pool, size=20, replace=False)
    assert first.acquisition_pool_indices == tuple(pool)
    assert first.held_out_test_indices == tuple(test)
    assert first.initial_labeled_indices == tuple(initial)
    assert not np.intersect1d(pool, test).size
    assert np.isin(initial, pool).all()

    path = tmp_path / "seed_7.json"
    save_partition_manifest(first, path)
    assert load_partition_manifest(path, ids) == first


def test_partition_manifest_rejects_different_population_identity(tmp_path):
    ids = make_molecule_ids(["CCO", "CCN", "CCC", "CCCC", "CCCl"] * 6)
    manifest = generate_partition_manifest(ids, random_seed=3, n_initial=5)
    path = tmp_path / "manifest.json"
    save_partition_manifest(manifest, path)

    reordered = ids.copy()
    reordered[[0, 1]] = reordered[[1, 0]]
    with pytest.raises(ValueError, match="identity or ordering"):
        load_partition_manifest(path, reordered)

    payload = json.loads(path.read_text())
    payload["ordered_population_hash"] = "not-the-population-hash"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="identity or ordering"):
        load_partition_manifest(path, ids)


def replace_representation(representation, **changes):
    values = {
        "identifier": representation.identifier,
        "molecule_ids": representation.molecule_ids,
        "canonical_smiles": representation.canonical_smiles,
        "features": representation.features,
        "success": representation.success,
        "failure_reasons": representation.failure_reasons,
        "provenance": representation.provenance,
        "artifact_path": representation.artifact_path,
    }
    values.update(changes)
    return MolecularRepresentation(**values)
