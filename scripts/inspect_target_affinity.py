import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.targets import load_target_table


def inspect_affinity(target):
    df = load_target_table(target)

    affinity = df["aff_nM_median"].dropna().to_numpy()

    print("\n" + "=" * 50)
    print(f"Target: {target}")
    print(f"Molecules: {len(df):,}")
    print("=" * 50)

    print("\nAffinity statistics (nM)")
    print("-" * 30)

    for p in [0, 10, 25, 50, 75, 90, 95, 99, 100]:
        value = np.percentile(affinity, p)
        print(f"P{p:02d}: {value:12.2f}")

    print("\nThreshold candidates")
    print("-" * 30)

    for threshold in [10, 50, 100, 500, 1000, 5000, 10000]:
        active_fraction = (affinity <= threshold).mean()
        print(f"{threshold:6d} nM : {active_fraction:6.1%} active")

    print("\nAffinity type composition")
    print("-" * 30)
    print(df["affinity_type"].value_counts(normalize=True).mul(100).round(2))

    output_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{target.lower()}_affinity_distribution.png"

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.hist(
        np.log10(affinity),
        bins=60,
        alpha=0.8,
    )

    for threshold in [100, 500, 1000, 5000]:
        ax.axvline(
            np.log10(threshold),
            linestyle="--",
            linewidth=1,
            label=f"{threshold} nM",
        )

    ax.set_xlabel("log10 affinity (nM)")
    ax.set_ylabel("Molecules")
    ax.set_title(f"{target} affinity distribution")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"\nSaved plot: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    args = parser.parse_args()

    inspect_affinity(args.target)


if __name__ == "__main__":
    main()