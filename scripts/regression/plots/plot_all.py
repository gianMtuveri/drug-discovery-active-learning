"""Generate the complete figure set for one benchmark."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate all benchmark figures.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    jobs = {
        "prediction": "plot_prediction_metrics.py",
        "discovery": "plot_discovery_metrics.py",
        "diagnostics": "plot_runtime.py",
        "ranking": "plot_model_ranking.py",
    }

    for subdir, script_name in jobs.items():
        command = [
            sys.executable,
            str(script_dir / script_name),
            "--input-dir",
            str(args.input_dir),
            "--output-dir",
            str(args.output_dir / subdir),
        ]
        subprocess.run(command, check=True)

    print(f"\nAll figures saved under: {args.output_dir}")


if __name__ == "__main__":
    main()
