
from src.molecular_data import load_bindingdb_parquet


def main():
    path = "data/processed/bindingdb_egfr_clean.parquet"

    df = load_bindingdb_parquet(path)

    print(df.shape)
    print(df.columns)
    print(df["label"].value_counts())
    print(df.head())


if __name__ == "__main__":
    main()