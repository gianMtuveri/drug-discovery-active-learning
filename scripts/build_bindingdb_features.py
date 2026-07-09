import numpy as np

from src.molecular_data import load_bindingdb_parquet
from src.fingerprints import featurize_smiles


def main():
    input_path = "data/processed/egfr_activity_dataset.parquet"

    output_X = "data/processed/bindingdb_egfr_morgan_X.npy"
    output_y = "data/processed/bindingdb_egfr_y.npy"

    df = load_bindingdb_parquet(input_path)

    X, valid_indices = featurize_smiles(
        df["canonical_smiles"].tolist(),
        radius=2,
        n_bits=2048,
    )

    y = df.iloc[valid_indices]["label"].astype(int).to_numpy()

    np.save(output_X, X)
    np.save(output_y, y)

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Active fraction:", y.mean())
    print("Saved:", output_X)
    print("Saved:", output_y)


if __name__ == "__main__":
    main()