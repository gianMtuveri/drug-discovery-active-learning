#!/usr/bin/env python3
"""Summarize the controlled Uni-Mol GP kernel diagnostic."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/acquisition_degeneracy/v051/gp_kernel_comparison"


def main() -> None:
    frame = pd.read_parquet(OUTPUT / "per_round_kernel_diagnostics.parquet")
    frame["tie_affected"] = frame.selected_tied_count > 0
    frame["parameter_at_bound"] = frame.parameters_at_bounds != "[]"
    seed_summary = frame.groupby(["kernel_family", "seed"], as_index=False).agg(
        complete_tie_rounds=("complete_tie", "sum"),
        tie_affected_rounds=("tie_affected", "sum"),
        tied_selected_molecules=("selected_tied_count", "sum"),
        selected_from_largest_tie_group=("selected_from_largest_tie_group", "sum"),
        optimizer_warning_rounds=("optimizer_warning", "sum"),
        parameter_bound_rounds=("parameter_at_bound", "sum"),
    )
    kernel_summary = seed_summary.groupby("kernel_family", as_index=False).agg(
        seeds_with_complete_tie=("complete_tie_rounds", lambda x: int((x > 0).sum())),
        complete_tie_rounds=("complete_tie_rounds", "sum"),
        seeds_with_tie_affected_round=("tie_affected_rounds", lambda x: int((x > 0).sum())),
        tie_affected_rounds=("tie_affected_rounds", "sum"),
        tied_selected_molecules=("tied_selected_molecules", "sum"),
        selected_from_largest_tie_group=("selected_from_largest_tie_group", "sum"),
        optimizer_warning_rounds=("optimizer_warning_rounds", "sum"),
        parameter_bound_rounds=("parameter_bound_rounds", "sum"),
    )
    kernel_summary["seeds_without_ties"] = kernel_summary.kernel_family.map(
        lambda name: seed_summary.loc[
            (seed_summary.kernel_family == name)
            & (seed_summary.tie_affected_rounds == 0),
            "seed",
        ].tolist()
    )
    distributions = {}
    for name, group in seed_summary.groupby("kernel_family"):
        distributions[name] = {
            "tie_affected_rounds_per_seed": {
                str(int(key)): int(value)
                for key, value in group.tie_affected_rounds.value_counts().sort_index().items()
            },
            "complete_tie_rounds_per_seed": {
                str(int(key)): int(value)
                for key, value in group.complete_tie_rounds.value_counts().sort_index().items()
            },
        }
    payload = {
        "purpose": "acquisition-surface diagnostic only; no performance comparison",
        "kernel_initialization": {
            "constant_value": 1.0,
            "length_scale": 1.0,
            "white_noise": 1.0,
            "rational_quadratic_alpha": 1.0,
            "bounds": [1e-5, 1e5],
        },
        "kernel_summary": kernel_summary.to_dict(orient="records"),
        "per_seed_distributions": distributions,
    }
    seed_summary.to_csv(OUTPUT / "per_seed_kernel_summary.csv", index=False)
    kernel_summary.to_csv(OUTPUT / "kernel_summary.csv", index=False)
    (OUTPUT / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
