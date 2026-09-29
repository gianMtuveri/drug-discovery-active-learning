import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, rdMolDescriptors


def compute_basic_descriptors(smiles_list):
    """
    Compute simple interpretable 2D molecular descriptors from SMILES.

    These descriptors do not require 3D conformers.
    """

    rows = []

    for idx, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            continue

        try:
            qed = QED.qed(mol)
        except (ArithmeticError, RuntimeError, ValueError):
            # Keep the failed value explicit so the preprocessing pipeline can
            # exclude it and report the reason; never substitute a prediction.
            qed = float("nan")

        rows.append(
            {
                "index": idx,
                "canonical_smiles": Chem.MolToSmiles(mol, canonical=True),
                "MolWt": Descriptors.MolWt(mol),
                "LogP": Descriptors.MolLogP(mol),
                "TPSA": rdMolDescriptors.CalcTPSA(mol),
                "HBD": rdMolDescriptors.CalcNumHBD(mol),
                "HBA": rdMolDescriptors.CalcNumHBA(mol),
                "RotatableBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
                "RingCount": rdMolDescriptors.CalcNumRings(mol),
                "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
                "QED": qed,
            }
        )

    return pd.DataFrame(rows)
