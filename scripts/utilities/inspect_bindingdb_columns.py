import argparse
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--zip-path",
        default="data/raw/BindingDB_All_202607_tsv.zip",
    )
    args = parser.parse_args()

    with zipfile.ZipFile(args.zip_path) as zf:
        print("Files inside zip:")
        print(zf.namelist())

        tsv_name = zf.namelist()[0]

        with zf.open(tsv_name) as f:
            header = f.readline().decode("utf-8").rstrip("\n").split("\t")

    print("\nColumns:")
    for i, col in enumerate(header):
        print(i, col)


if __name__ == "__main__":
    main()