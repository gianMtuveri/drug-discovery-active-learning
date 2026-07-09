import pandas as pd


def load_bindingdb_parquet(path):
    """
    Load cleaned BindingDB data from parquet.
    """

    df = pd.read_parquet(path)

    required = ["canonical_smiles", "label"]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=required).copy()

    return df