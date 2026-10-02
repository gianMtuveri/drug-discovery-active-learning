from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from rdkit import Chem


SCRIPT = Path(__file__).parents[1] / "scripts" / "representations" / "run_embedding_pilot.py"
SPEC = importlib.util.spec_from_file_location("run_embedding_pilot", SCRIPT)
PILOT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = PILOT
SPEC.loader.exec_module(PILOT)


def _selection_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "original_index": np.arange(30),
        "heavy_atom_count": np.arange(10, 40),
        "smi_ted_token_count": np.arange(20, 80, 2),
        "smiles_length": np.arange(30, 90, 2),
        "has_stereochemistry": [value % 3 == 0 for value in range(30)],
    })


def test_pilot_selection_is_deterministic_and_structure_only():
    frame = _selection_frame()
    first = PILOT.select_pilot(frame, pilot_size=20)
    second = PILOT.select_pilot(frame.assign(affinity=np.arange(30)[::-1]), pilot_size=20)
    assert first == second
    assert len(first) == 20
    assert min(frame.original_index) in first
    assert max(frame.original_index) in first


def test_unimol_atom_limit_is_rejected_before_geometry():
    smiles = ".".join(["C"] * (PILOT.UNIMOL_MAX_ATOMS + 1))
    status, reason, code, mol, coordinates = PILOT.generate_unimol_geometry(smiles)
    assert status == "atom_limit_exceeded"
    assert reason == "257>256"
    assert code is None and mol is None and coordinates is None


def test_coordinate_validation_rejects_zero_coordinates():
    mol = Chem.AddHs(Chem.MolFromSmiles("CC"))
    conformer = Chem.Conformer(mol.GetNumAtoms())
    conformer.Set3D(True)
    mol.AddConformer(conformer)
    valid, reason = PILOT.coordinates_are_valid(
        mol, np.zeros((mol.GetNumAtoms(), 3), dtype=np.float32)
    )
    assert not valid
    assert reason == "zero_or_collapsed_coordinates"


def test_deterministic_conformer_generation():
    first = PILOT.generate_unimol_geometry("C[C@H](O)C(=O)O")
    second = PILOT.generate_unimol_geometry("C[C@H](O)C(=O)O")
    assert first[0] in {"optimized", "mmff_unavailable", "mmff_nonconverged"}
    assert first[0:3] == second[0:3]
    np.testing.assert_allclose(first[4], second[4], atol=0.0, rtol=0.0)


def test_unimol_geometry_removes_deuterium_without_mutating_source_structure():
    smiles = "[2H]C([2H])([2H])CO"
    original = Chem.MolFromSmiles(smiles)
    original_smiles = Chem.MolToSmiles(original, canonical=True, isomericSmiles=True)
    expected_non_h_symbols = [
        atom.GetSymbol() for atom in original.GetAtoms() if atom.GetAtomicNum() != 1
    ]

    status, reason, _, no_h, coordinates = PILOT.generate_unimol_geometry(smiles)

    assert status in {"optimized", "mmff_unavailable", "mmff_nonconverged"}, reason
    assert Chem.MolToSmiles(original, canonical=True, isomericSmiles=True) == original_smiles
    assert sum(atom.GetAtomicNum() == 1 for atom in original.GetAtoms()) == 3
    assert no_h is not None and coordinates is not None
    assert all(atom.GetAtomicNum() != 1 for atom in no_h.GetAtoms())
    assert [atom.GetSymbol() for atom in no_h.GetAtoms()] == expected_non_h_symbols
    assert coordinates.shape == (len(expected_non_h_symbols), 3)
    assert np.isfinite(coordinates).all()
    valid, validation_reason = PILOT.coordinates_are_valid(no_h, coordinates)
    assert valid, validation_reason
