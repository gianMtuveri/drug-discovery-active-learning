import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors


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

        rows.append(
            {
                "index": idx,
                "MolWt": Descriptors.MolWt(mol),
                "LogP": Descriptors.MolLogP(mol),
                "TPSA": rdMolDescriptors.CalcTPSA(mol),
                "HBD": rdMolDescriptors.CalcNumHBD(mol),
                "HBA": rdMolDescriptors.CalcNumHBA(mol),
                "RotatableBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
                "RingCount": rdMolDescriptors.CalcNumRings(mol),
                "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
            }
        )

    return pd.DataFrame(rows)