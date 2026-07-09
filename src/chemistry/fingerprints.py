import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


def smiles_to_morgan_fingerprint(
    smiles,
    radius=2,
    n_bits=2048,
):
    """
    Convert one SMILES string into a Morgan fingerprint.
    """

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=n_bits,
    )

    fingerprint = generator.GetFingerprint(mol)

    array = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fingerprint, array)

    return array


def featurize_smiles(
    smiles_list,
    radius=2,
    n_bits=2048,
):
    """
    Convert a list of SMILES into Morgan fingerprints.
    Invalid SMILES are skipped.
    """

    fingerprints = []
    valid_indices = []

    for idx, smiles in enumerate(smiles_list):
        fp = smiles_to_morgan_fingerprint(
            smiles,
            radius=radius,
            n_bits=n_bits,
        )

        if fp is None:
            continue

        fingerprints.append(fp)
        valid_indices.append(idx)

    X = np.vstack(fingerprints)

    return X, np.array(valid_indices, dtype=int)