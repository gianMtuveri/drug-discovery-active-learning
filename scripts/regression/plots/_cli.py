from __future__ import annotations

import argparse
from pathlib import Path


def parse_single_benchmark_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Benchmark directory containing round_summary.csv and final_round_summary.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory in which figures will be written.",
    )
    return parser.parse_args()
