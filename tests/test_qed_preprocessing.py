import json

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import QED

from scripts.preprocessing.prepare_target import prepare_target
from src.chemistry.descriptors import compute_basic_descriptors
from src.chemistry import descriptors as descriptor_module
from src.data.targets import validate_target_alignment


def test_compute_basic_descriptors_adds_rdkit_qed_and_skips_invalid_smiles():
    smiles = ["CCO", "not-a-smiles"]

    descriptors = compute_basic_descriptors(smiles)

    assert descriptors["index"].tolist() == [0]
    assert descriptors["canonical_smiles"].tolist() == ["CCO"]
    assert descriptors.loc[0, "QED"] == pytest.approx(QED.qed(Chem.MolFromSmiles("CCO")))
    assert np.isfinite(descriptors["QED"]).all()
    assert descriptors["QED"].between(0, 1, inclusive="both").all()


def test_prepare_target_excludes_invalid_rows_and_preserves_alignment(tmp_path):
    master = tmp_path / "master.parquet"
    registry = tmp_path / "targets.yaml"
    output = tmp_path / "targets"
    pd.DataFrame(
        {
            "Target Name": ["Test target"] * 4,
            "canonical_smiles": ["CCO", "invalid", "CCN", "CCC"],
            "aff_nM_median": [10.0, 20.0, np.nan, 0.0],
        }
    ).to_parquet(master, index=False)
    registry.write_text(
        "TEST:\n  query: Test target\n  match: exact\n  activity_threshold_nM: 100\n"
    )

    prepare_target("TEST", master, registry, output, n_bits=64)

    target_dir = output / "TEST"
    lengths = validate_target_alignment("TEST", output)
    assert set(lengths.values()) == {1}
    descriptors = pd.read_parquet(target_dir / "descriptors.parquet")
    assert descriptors["canonical_smiles"].tolist() == ["CCO"]
    assert np.isfinite(descriptors["QED"]).all()
    assert descriptors["QED"].between(0, 1, inclusive="both").all()
    metadata = json.loads((target_dir / "metadata.json").read_text())
    assert metadata["exclusions"] == {
        "invalid_structure_or_fingerprint": 1,
        "non_finite_or_non_positive_affinity": 2,
        "invalid_qed": 0,
    }


def test_alignment_validation_detects_molecular_reordering(tmp_path):
    target_dir = tmp_path / "TEST"
    target_dir.mkdir()
    smiles = np.array(["CCO", "CCN"])
    np.savez_compressed(target_dir / "X_morgan.npz", X=np.zeros((2, 8)), canonical_smiles=smiles)
    np.save(target_dir / "affinity.npy", np.array([8.0, 7.0]))
    pd.DataFrame({"canonical_smiles": smiles[::-1], "QED": [0.4, 0.5]}).to_parquet(
        target_dir / "descriptors.parquet", index=False
    )
    pd.DataFrame({"canonical_smiles": smiles}).to_parquet(
        target_dir / "clean.parquet", index=False
    )

    with pytest.raises(ValueError, match="order differs"):
        validate_target_alignment("TEST", tmp_path)


def test_prepare_target_excludes_non_finite_qed(tmp_path, monkeypatch):
    original_qed = descriptor_module.QED.qed

    def qed_with_one_failure(mol):
        if Chem.MolToSmiles(mol) == "CCN":
            return float("nan")
        return original_qed(mol)

    monkeypatch.setattr(descriptor_module.QED, "qed", qed_with_one_failure)
    master = tmp_path / "master.parquet"
    registry = tmp_path / "targets.yaml"
    pd.DataFrame(
        {
            "Target Name": ["Test target", "Test target"],
            "canonical_smiles": ["CCO", "CCN"],
            "aff_nM_median": [10.0, 20.0],
        }
    ).to_parquet(master, index=False)
    registry.write_text("TEST:\n  query: Test target\n  match: exact\n")

    prepare_target("TEST", master, registry, tmp_path / "targets", n_bits=64)

    metadata = json.loads(
        (tmp_path / "targets" / "TEST" / "metadata.json").read_text()
    )
    assert metadata["exclusions"]["invalid_qed"] == 1
    assert validate_target_alignment("TEST", tmp_path / "targets")["X_morgan"] == 1
