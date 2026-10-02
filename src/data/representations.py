"""Molecular representation artifacts and alignment validation.

Representation rows always remain aligned to the source molecular population.
Failed encodings are represented by ``success=False`` and an explicit reason;
they are never removed implicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.data.targets import get_target_dir, load_target_descriptors


REPRESENTATION_SCHEMA_VERSION = 1
REPRESENTATION_LOADER_VERSION = "v051.1"

RDKIT_DESCRIPTOR_COLUMNS = (
    "MolWt",
    "LogP",
    "TPSA",
    "HBD",
    "HBA",
    "RotatableBonds",
    "RingCount",
    "FractionCSP3",
)


@dataclass(frozen=True)
class MolecularRepresentation:
    """One feature row and encoding status per source molecule."""

    identifier: str
    molecule_ids: np.ndarray
    canonical_smiles: np.ndarray
    features: np.ndarray
    success: np.ndarray
    failure_reasons: np.ndarray
    provenance: Mapping[str, Any]
    artifact_path: Path | None = None

    @property
    def feature_dimension(self) -> int:
        return int(self.features.shape[1])

    @property
    def dtype(self) -> np.dtype:
        return self.features.dtype


@dataclass(frozen=True)
class CommonPopulation:
    """Intersection of successful encodings in source-population order."""

    molecule_ids: np.ndarray
    canonical_smiles: np.ndarray
    inclusion_mask: np.ndarray
    exclusion_reasons: tuple[Mapping[str, str], ...]
    representation_identifiers: tuple[str, ...]
    original_size: int
    common_size: int
    ordered_population_hash: str


def make_molecule_ids(canonical_smiles: Sequence[str]) -> np.ndarray:
    """Build stable IDs while retaining repeated canonical-SMILES rows."""
    occurrences: dict[str, int] = {}
    identifiers: list[str] = []
    for value in canonical_smiles:
        if value is None or pd.isna(value) or not str(value).strip():
            raise ValueError("Canonical SMILES cannot be missing or empty")
        smiles = str(value)
        occurrence = occurrences.get(smiles, 0)
        occurrences[smiles] = occurrence + 1
        digest = sha256(
            f"canonical_smiles\0{smiles}\0occurrence\0{occurrence}".encode("utf-8")
        ).hexdigest()
        identifiers.append(f"mol-{digest}")
    return np.asarray(identifiers, dtype=str)


def ordered_population_hash(molecule_ids: Sequence[str]) -> str:
    """Hash an ordered identity sequence without ambiguous concatenation."""
    payload = json.dumps(
        [str(value) for value in molecule_ids],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def representation_artifact_dir(
    target: str,
    base_dir: str | Path = "data/processed/targets",
) -> Path:
    """Return the local, Git-ignored representation artifact directory."""
    return get_target_dir(target, base_dir) / "representations"


def _base_provenance(
    *,
    identifier: str,
    molecule_ids: np.ndarray,
    features: np.ndarray,
    source_path: Path,
    generation_parameters: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": REPRESENTATION_SCHEMA_VERSION,
        "representation": identifier,
        "ordered_population_hash": ordered_population_hash(molecule_ids),
        "feature_shape": list(features.shape),
        "feature_dtype": str(features.dtype),
        "loader_version": REPRESENTATION_LOADER_VERSION,
        "generation_parameters": dict(generation_parameters),
        "source_artifact": str(source_path),
        "source_artifact_sha256": file_sha256(source_path),
        # Future extractors may add checkpoint, tokenizer/dictionary,
        # conformer, and environment dictionaries without changing schema.
    }


def load_morgan_representation(
    target: str,
    base_dir: str | Path = "data/processed/targets",
) -> MolecularRepresentation:
    """Load the existing Morgan matrix without regeneration or coercion."""
    path = get_target_dir(target, base_dir) / "X_morgan.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing Morgan artifact: {path}")
    with np.load(path) as artifact:
        if "X" not in artifact.files or "canonical_smiles" not in artifact.files:
            raise ValueError("Morgan artifact must contain X and canonical_smiles")
        features = artifact["X"]
        smiles = artifact["canonical_smiles"].astype(str)
    molecule_ids = make_molecule_ids(smiles)
    success = np.ones(len(smiles), dtype=bool)
    reasons = np.full(len(smiles), "", dtype=object)
    provenance = _base_provenance(
        identifier="morgan",
        molecule_ids=molecule_ids,
        features=features,
        source_path=path,
        generation_parameters={"radius": 2, "n_bits": 2048, "regenerated": False},
    )
    representation = MolecularRepresentation(
        identifier="morgan",
        molecule_ids=molecule_ids,
        canonical_smiles=smiles,
        features=features,
        success=success,
        failure_reasons=reasons,
        provenance=provenance,
        artifact_path=path,
    )
    validate_representation(representation, expected_dimension=2048)
    return representation


def load_rdkit_descriptor_representation(
    target: str,
    base_dir: str | Path = "data/processed/targets",
    *,
    reference: MolecularRepresentation | None = None,
) -> MolecularRepresentation:
    """Load exactly the approved eight existing RDKit descriptor columns."""
    path = get_target_dir(target, base_dir) / "descriptors.parquet"
    descriptors = load_target_descriptors(target, base_dir)
    required = {"canonical_smiles", *RDKIT_DESCRIPTOR_COLUMNS}
    missing = sorted(required.difference(descriptors.columns))
    if missing:
        raise ValueError(f"Descriptor artifact is missing columns: {missing}")

    if descriptors["canonical_smiles"].isna().any():
        raise ValueError("Descriptor artifact contains missing canonical SMILES")
    smiles = descriptors["canonical_smiles"].astype(str).to_numpy()
    molecule_ids = make_molecule_ids(smiles)
    # This selection both fixes order and excludes QED/labels by construction.
    frame = descriptors.loc[:, list(RDKIT_DESCRIPTOR_COLUMNS)]
    if not all(pd.api.types.is_numeric_dtype(frame[column]) for column in frame):
        raise ValueError("All approved RDKit descriptor columns must be numeric")
    features = frame.to_numpy(copy=True)
    success = np.ones(len(smiles), dtype=bool)
    reasons = np.full(len(smiles), "", dtype=object)
    provenance = _base_provenance(
        identifier="rdkit_descriptors",
        molecule_ids=molecule_ids,
        features=features,
        source_path=path,
        generation_parameters={
            "descriptor_columns": list(RDKIT_DESCRIPTOR_COLUMNS),
            "qed_in_surrogate_features": False,
            "recomputed": False,
        },
    )
    representation = MolecularRepresentation(
        identifier="rdkit_descriptors",
        molecule_ids=molecule_ids,
        canonical_smiles=smiles,
        features=features,
        success=success,
        failure_reasons=reasons,
        provenance=provenance,
        artifact_path=path,
    )
    validate_representation(
        representation,
        reference=reference,
        expected_dimension=len(RDKIT_DESCRIPTOR_COLUMNS),
    )
    return representation


def load_frozen_embedding_representation(
    identifier: str,
    artifact_dir: str | Path,
    *,
    expected_dimension: int,
    reference: MolecularRepresentation | None = None,
) -> MolecularRepresentation:
    """Load a finalized aligned embedding artifact using its stored success mask."""
    artifact_dir = Path(artifact_dir)
    embeddings_path = artifact_dir / "embeddings.npy"
    mask_path = artifact_dir / "success_mask.npy"
    ids_path = artifact_dir / "molecule_ids.npy"
    audit_path = artifact_dir / "extraction_audit.parquet"
    manifest_path = artifact_dir / "manifest.json"
    for path in (embeddings_path, mask_path, ids_path, audit_path, manifest_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing finalized representation artifact: {path}")

    features = np.load(embeddings_path, mmap_mode="r")
    success = np.load(mask_path).astype(bool, copy=False)
    molecule_ids = np.load(ids_path).astype(str)
    audit = pd.read_parquet(audit_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_audit = {"molecule_id", "canonical_smiles", "status", "failure_reason"}
    missing = sorted(required_audit.difference(audit.columns))
    if missing:
        raise ValueError(f"Finalized representation audit is missing columns: {missing}")
    if not np.array_equal(molecule_ids, audit.molecule_id.astype(str).to_numpy()):
        raise ValueError("Stored molecule IDs differ from extraction-audit ordering")
    if len(success) != len(features):
        raise ValueError("Stored success mask differs from embedding row count")
    finite_rows = np.isfinite(features).all(axis=1)
    nan_rows = np.isnan(features).all(axis=1)
    if not np.array_equal(success, finite_rows):
        raise ValueError("Stored success mask differs from finite embedding rows")
    if not np.all(nan_rows[~success]):
        raise ValueError("Failed embedding rows are not all-NaN")

    raw_reasons = audit.failure_reason.fillna("").astype(str).to_numpy()
    statuses = audit.status.astype(str).to_numpy()
    reasons = np.where(success, "", np.where(raw_reasons != "", raw_reasons, statuses))
    provenance = {
        "schema_version": REPRESENTATION_SCHEMA_VERSION,
        "representation": identifier,
        "ordered_population_hash": ordered_population_hash(molecule_ids),
        "feature_shape": list(features.shape),
        "feature_dtype": str(features.dtype),
        "loader_version": REPRESENTATION_LOADER_VERSION,
        "generation_parameters": {
            "loaded_from_finalized_artifact": True,
            "artifact_manifest_schema_version": manifest.get("schema_version"),
        },
        "source_artifact": str(artifact_dir),
        "source_manifest": manifest,
        "artifact_checksums": {
            path.name: file_sha256(path)
            for path in (embeddings_path, mask_path, ids_path, audit_path, manifest_path)
        },
    }
    representation = MolecularRepresentation(
        identifier=identifier,
        molecule_ids=molecule_ids,
        canonical_smiles=audit.canonical_smiles.astype(str).to_numpy(),
        features=features,
        success=success,
        failure_reasons=np.asarray(reasons, dtype=object),
        provenance=provenance,
        artifact_path=artifact_dir,
    )
    validate_representation(
        representation, reference=reference, expected_dimension=expected_dimension
    )
    return representation


def validate_representation(
    representation: MolecularRepresentation,
    *,
    reference: MolecularRepresentation | None = None,
    expected_dimension: int | None = None,
) -> None:
    """Validate row alignment, statuses, features, and core provenance."""
    if not representation.identifier:
        raise ValueError("Representation identifier must be non-empty")

    ids = np.asarray(representation.molecule_ids)
    smiles = np.asarray(representation.canonical_smiles)
    features = np.asarray(representation.features)
    success = np.asarray(representation.success)
    reasons = np.asarray(representation.failure_reasons, dtype=object)
    if ids.ndim != 1 or smiles.ndim != 1 or success.ndim != 1 or reasons.ndim != 1:
        raise ValueError("Representation identities and statuses must be one-dimensional")
    if features.ndim != 2:
        raise ValueError("Representation features must be two-dimensional")
    lengths = {len(ids), len(smiles), len(features), len(success), len(reasons)}
    if len(lengths) != 1:
        raise ValueError("Representation rows, identities, and statuses are not aligned")
    if len(ids) == 0:
        raise ValueError("Representation must contain at least one molecule")
    if any(not str(value) for value in ids) or any(not str(value) for value in smiles):
        raise ValueError("Molecular identities and canonical SMILES cannot be missing")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Duplicate molecular identities found")
    if success.dtype != np.bool_:
        raise ValueError("Eligibility success mask must have boolean dtype")
    if expected_dimension is not None and features.shape[1] != expected_dimension:
        raise ValueError(
            f"Expected feature dimension {expected_dimension}, got {features.shape[1]}"
        )
    if not np.issubdtype(features.dtype, np.number):
        raise ValueError("Representation features must be numeric")
    if not np.isfinite(features[success]).all():
        raise ValueError("Successful representation rows contain non-finite features")

    normalized_reasons = np.asarray([str(reason).strip() for reason in reasons])
    if np.any(success & (normalized_reasons != "")):
        raise ValueError("Successful rows cannot have failure reasons")
    if np.any(~success & (normalized_reasons == "")):
        raise ValueError("Every failed representation row requires an explicit reason")

    required_provenance = {
        "schema_version",
        "representation",
        "ordered_population_hash",
        "feature_shape",
        "feature_dtype",
        "loader_version",
        "generation_parameters",
    }
    missing_provenance = sorted(required_provenance.difference(representation.provenance))
    if missing_provenance:
        raise ValueError(f"Representation provenance is missing: {missing_provenance}")
    expected_hash = ordered_population_hash(ids)
    checks = {
        "representation": representation.identifier,
        "ordered_population_hash": expected_hash,
        "feature_shape": list(features.shape),
        "feature_dtype": str(features.dtype),
    }
    for key, expected in checks.items():
        if representation.provenance[key] != expected:
            raise ValueError(f"Representation provenance mismatch for {key}")

    if reference is not None:
        if not np.array_equal(ids, reference.molecule_ids):
            raise ValueError("Molecular identity order differs from reference representation")
        if not np.array_equal(smiles, reference.canonical_smiles):
            raise ValueError("Canonical SMILES order differs from reference representation")


def build_common_population(
    representations: Sequence[MolecularRepresentation],
) -> CommonPopulation:
    """Intersect success masks without consulting labels, QED, or affinity."""
    if not representations:
        raise ValueError("At least one representation is required")
    identifiers = [representation.identifier for representation in representations]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Representation identifiers must be unique")

    reference = representations[0]
    validate_representation(reference)
    for representation in representations[1:]:
        validate_representation(representation, reference=reference)

    inclusion = np.logical_and.reduce(
        [np.asarray(representation.success) for representation in representations]
    )
    exclusions: list[Mapping[str, str]] = []
    for row in range(len(inclusion)):
        exclusions.append(
            {
                representation.identifier: str(representation.failure_reasons[row])
                for representation in representations
                if not representation.success[row]
            }
        )

    return CommonPopulation(
        molecule_ids=np.asarray(reference.molecule_ids).copy(),
        canonical_smiles=np.asarray(reference.canonical_smiles).copy(),
        inclusion_mask=inclusion,
        exclusion_reasons=tuple(exclusions),
        representation_identifiers=tuple(identifiers),
        original_size=len(inclusion),
        common_size=int(inclusion.sum()),
        ordered_population_hash=ordered_population_hash(reference.molecule_ids),
    )
