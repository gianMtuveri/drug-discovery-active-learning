from src.molecular_data import load_bindingdb_parquet
from src.descriptors import compute_basic_descriptors


def main():
    input_path = "data/processed/egfr_activity_dataset.parquet"
    output_path = "data/processed/bindingdb_egfr_descriptors.parquet"

    df = load_bindingdb_parquet(input_path)

    descriptors = compute_basic_descriptors(
        df["canonical_smiles"].tolist()
    )

    descriptors["label"] = df.iloc[descriptors["index"]]["label"].to_numpy()

    descriptors.to_parquet(output_path, index=False)

    print(descriptors.shape)
    print(descriptors.head())
    print("Saved:", output_path)


if __name__ == "__main__":
    main()