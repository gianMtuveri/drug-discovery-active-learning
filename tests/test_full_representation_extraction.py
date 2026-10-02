from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.representations import extract_full_frozen_representations as extraction


def _population() -> pd.DataFrame:
    return pd.DataFrame({
        "original_index": [0, 1, 2],
        "molecule_id": ["mol-a", "mol-b", "mol-c"],
        "canonical_smiles": ["C", "CC", "CCC"],
    })


def test_aligned_artifact_preserves_failed_nan_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(extraction, "EXPECTED_ROWS", 3)
    population = _population()
    audit = population.assign(
        status=["success", "token_limit_exceeded", "success"],
        failure_reason=["", "203>202", ""],
    )
    features = np.asarray([[1, 2], [np.nan, np.nan], [3, 4]], dtype=np.float32)
    success = np.asarray([True, False, True])

    extraction._finalize_representation(
        tmp_path, population, audit, features, success, 2, {"representation": "synthetic"}
    )

    np.testing.assert_array_equal(np.load(tmp_path / "success_mask.npy"), success)
    np.testing.assert_array_equal(
        np.load(tmp_path / "molecule_ids.npy"), population.molecule_id.to_numpy()
    )


def test_success_mask_must_match_finite_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(extraction, "EXPECTED_ROWS", 3)
    population = _population()
    audit = population.assign(status=["success"] * 3, failure_reason=[""] * 3)
    features = np.ones((3, 2), dtype=np.float32)
    features[1] = np.nan

    with pytest.raises(ValueError, match="Success mask"):
        extraction._finalize_representation(
            tmp_path, population, audit, features, np.ones(3, dtype=bool), 2, {}
        )


def test_resume_alignment_detects_reordering():
    population = _population()
    reordered = population.iloc[[1, 0, 2]].reset_index(drop=True)
    with pytest.raises(ValueError, match="misaligned"):
        extraction._assert_population_alignment(reordered, population)
