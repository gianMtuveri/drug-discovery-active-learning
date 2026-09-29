import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr


def evaluate_regression(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        pearson = np.nan
    else:
        pearson = pearsonr(y_true, y_pred).statistic

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson": pearson,
    }


def top_k_mean_affinity(y_selected, k=20):
    k = min(k, len(y_selected))
    return np.mean(np.sort(y_selected)[-k:])


def best_affinity(y_selected):
    return np.max(y_selected)


def top_k_enrichment(y_selected, y_all, k=100):
    global_top_k_threshold = np.sort(y_all)[-k]

    n_selected_top_k = np.sum(y_selected >= global_top_k_threshold)

    return n_selected_top_k / k


def joint_discovery_mask(
    affinity_nm,
    qed,
    *,
    activity_threshold_nm=500.0,
    qed_threshold=0.6,
):
    """Identify experimentally active, QED-qualified molecules.

    Activity follows the preprocessing convention exactly: lower nM values
    are more potent and the threshold is inclusive.
    """
    affinity_nm = np.asarray(affinity_nm, dtype=float)
    qed = np.asarray(qed, dtype=float)
    if affinity_nm.ndim != 1 or qed.ndim != 1 or len(affinity_nm) != len(qed):
        raise ValueError("Affinity and QED must be aligned one-dimensional arrays.")
    return (affinity_nm <= activity_threshold_nm) & (qed >= qed_threshold)


def bemis_murcko_scaffolds(canonical_smiles):
    """Return one canonical Bemis-Murcko scaffold SMILES per molecule.

    RDKit returns an empty scaffold for acyclic molecules.  Empty scaffolds
    are retained in the aligned array but are not counted as scaffolds.
    """
    scaffolds = []
    for smiles in canonical_smiles:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            raise ValueError(f"Invalid canonical SMILES: {smiles}")
        # Bemis-Murcko identity does not use stereochemistry.  Removing it on
        # a copy also avoids RDKit canonicalization failures for otherwise
        # valid molecules with stereo annotations on bonds removed by Murcko.
        mol = Chem.Mol(mol)
        Chem.RemoveStereochemistry(mol)
        scaffolds.append(MurckoScaffold.MurckoScaffoldSmiles(mol=mol))
    return np.asarray(scaffolds, dtype=str)


def count_distinct_scaffolds(scaffolds):
    """Count distinct non-empty Bemis-Murcko scaffold SMILES."""
    return len({str(scaffold) for scaffold in scaffolds if str(scaffold)})


def mean_pairwise_tanimoto_distance(fingerprints):
    """Mean ``1 - Tanimoto similarity`` over all unordered molecule pairs.

    Diversity is undefined for fewer than two molecules and is reported as
    ``NaN``.  The vectorized upper triangle avoids constructing RDKit pair
    objects and remains small because campaigns acquire at most 200 molecules.
    """
    fingerprints = np.asarray(fingerprints)
    if fingerprints.ndim != 2:
        raise ValueError("Fingerprints must be a two-dimensional array.")
    if len(fingerprints) < 2:
        return np.nan

    binary = fingerprints.astype(bool, copy=False)
    intersections = binary.astype(np.uint16) @ binary.T.astype(np.uint16)
    counts = binary.sum(axis=1)
    unions = counts[:, None] + counts[None, :] - intersections
    rows, columns = np.triu_indices(len(binary), k=1)
    similarities = intersections[rows, columns] / unions[rows, columns]
    return float(np.mean(1.0 - similarities))


def cumulative_curve_auc(measurements, cumulative_discoveries):
    """Trapezoidal AUC using newly acquired measurement count as x-axis."""
    x = np.asarray(measurements, dtype=float)
    y = np.asarray(cumulative_discoveries, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("Curve coordinates must be aligned one-dimensional arrays.")
    if len(x) < 2:
        return 0.0
    if np.any(np.diff(x) < 0):
        raise ValueError("Measurement counts must be non-decreasing.")
    widths = np.diff(x)
    return float(np.sum(widths * (y[:-1] + y[1:]) / 2.0))


def evaluate_acquired_molecules(
    *,
    acquired_indices,
    affinity_nm,
    qed,
    fingerprints,
    scaffolds,
    activity_threshold_nm=500.0,
    qed_threshold=0.6,
):
    """Calculate experimental discovery endpoints for newly acquired molecules."""
    acquired_indices = np.asarray(acquired_indices, dtype=int)
    affinity_nm = np.asarray(affinity_nm, dtype=float)
    qed = np.asarray(qed, dtype=float)
    fingerprints = np.asarray(fingerprints)
    scaffolds = np.asarray(scaffolds)

    if len(np.unique(acquired_indices)) != len(acquired_indices):
        raise ValueError("Acquired molecule indices must be distinct.")
    n_molecules = len(affinity_nm)
    if not (len(qed) == len(fingerprints) == len(scaffolds) == n_molecules):
        raise ValueError("Evaluation arrays are not molecularly aligned.")

    selected_affinity = affinity_nm[acquired_indices]
    selected_qed = qed[acquired_indices]
    active = selected_affinity <= activity_threshold_nm
    joint = joint_discovery_mask(
        selected_affinity,
        selected_qed,
        activity_threshold_nm=activity_threshold_nm,
        qed_threshold=qed_threshold,
    )
    qualifying_indices = acquired_indices[joint]

    return {
        "affinity_discovery_count": int(active.sum()),
        "joint_discovery_count": int(joint.sum()),
        "joint_scaffold_count": count_distinct_scaffolds(scaffolds[qualifying_indices]),
        "joint_diversity": mean_pairwise_tanimoto_distance(
            fingerprints[qualifying_indices]
        ),
        "joint_qualifying_indices": qualifying_indices,
    }
