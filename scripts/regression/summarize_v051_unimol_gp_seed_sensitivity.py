#!/usr/bin/env python3
"""Summarize the completed Uni-Mol GP seed-sensitivity diagnostic."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/acquisition_degeneracy/v051/gp_seed_sensitivity"


def main() -> None:
    frame = pd.read_parquet(OUTPUT / "per_round_seed_diagnostics.parquet")
    frame["tie_affected"] = frame.selected_tied_count > 0
    lengths = np.sort(frame.length_scale.unique())
    gaps = np.diff(lengths)
    split = int(np.argmax(gaps))
    lower_mode_max = float(lengths[split])
    upper_mode_min = float(lengths[split + 1])
    frame["lower_length_scale_mode"] = frame.length_scale <= lower_mode_max

    seed_records = []
    for seed, group in frame.groupby("seed"):
        group = group.sort_values("round")
        tied = group.tie_affected.to_numpy()
        tied_rounds = group.loc[group.tie_affected, "round"]
        first = int(tied_rounds.min()) if len(tied_rounds) else None
        recovered = False
        recurred = False
        if first is not None:
            seen_recovery = False
            for value in group.loc[group["round"] > first, "tie_affected"]:
                if not value:
                    recovered = True
                    seen_recovery = True
                elif seen_recovery:
                    recurred = True
        seed_records.append(
            {
                "seed": int(seed),
                "complete_tie_rounds": int(group.complete_tie.sum()),
                "tie_affected_rounds": int(group.tie_affected.sum()),
                "tied_selected_molecules": int(group.selected_tied_count.sum()),
                "selected_from_largest_tie_group": int(
                    group.selected_from_largest_tie_group.sum()
                ),
                "first_tie_affected_round": first,
                "lower_length_scale_mode_rounds": int(
                    group.lower_length_scale_mode.sum()
                ),
                "recovered_after_tie_affected_round": recovered,
                "recurred_after_recovery": recurred,
                "persistent_from_first_tie": bool(
                    first is not None
                    and group.loc[group["round"] >= first, "tie_affected"].all()
                ),
                "tie_state_transitions": int(np.sum(tied[1:] != tied[:-1])),
            }
        )
    seeds = pd.DataFrame(seed_records)
    rounds = frame.groupby("round", as_index=False).agg(
        tie_affected_seeds=("tie_affected", "sum"),
        complete_tie_seeds=("complete_tie", "sum"),
        tied_selected_molecules=("selected_tied_count", "sum"),
    )

    def describe(mask: pd.Series) -> dict[str, float | int]:
        values = frame.loc[mask, "length_scale"]
        return {
            "n": int(len(values)),
            "minimum": float(values.min()),
            "q25": float(values.quantile(0.25)),
            "median": float(values.median()),
            "mean": float(values.mean()),
            "q75": float(values.quantile(0.75)),
            "maximum": float(values.max()),
        }

    payload = {
        "classification": {
            "tie_affected": "selected_tied_count > 0",
            "complete_tie": "normalized UCB has exactly one unique value",
            "lower_length_scale_mode": {
                "method": "largest observed gap in the continuous fitted length-scale distribution",
                "lower_mode_max": lower_mode_max,
                "upper_mode_min": upper_mode_min,
                "gap": upper_mode_min - lower_mode_max,
            },
        },
        "aggregate": {
            "seeds_with_complete_tie": int((seeds.complete_tie_rounds > 0).sum()),
            "complete_tie_rounds": int(frame.complete_tie.sum()),
            "seeds_with_tie_affected_round": int(
                (seeds.tie_affected_rounds > 0).sum()
            ),
            "tie_affected_rounds": int(frame.tie_affected.sum()),
            "tied_selected_molecules": int(frame.selected_tied_count.sum()),
            "total_selected_molecules": 4000,
            "selected_from_largest_tie_group": int(
                frame.selected_from_largest_tie_group.sum()
            ),
            "seeds_without_tie_affected_round": seeds.loc[
                seeds.tie_affected_rounds == 0, "seed"
            ].tolist(),
            "persistent_seeds": seeds.loc[seeds.persistent_from_first_tie, "seed"].tolist(),
            "recurrent_seeds": seeds.loc[seeds.recurred_after_recovery, "seed"].tolist(),
            "optimizer_warning_rounds": int(frame.optimizer_warning.sum()),
            "parameter_bound_rounds": int(
                frame.parameters_at_bounds.ne("[]").sum()
            ),
        },
        "length_scale": {
            "no_selected_ties": describe(~frame.tie_affected),
            "selected_ties": describe(frame.tie_affected),
            "complete_ties": describe(frame.complete_tie),
        },
        "tie_affected_rounds_per_seed_distribution": {
            str(int(key)): int(value)
            for key, value in seeds.tie_affected_rounds.value_counts().sort_index().items()
        },
        "complete_ties_per_seed_distribution": {
            str(int(key)): int(value)
            for key, value in seeds.complete_tie_rounds.value_counts().sort_index().items()
        },
    }
    seeds.to_csv(OUTPUT / "per_seed_summary.csv", index=False)
    rounds.to_csv(OUTPUT / "round_frequency.csv", index=False)
    frame.to_parquet(OUTPUT / "per_round_seed_diagnostics.parquet", index=False)
    (OUTPUT / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
