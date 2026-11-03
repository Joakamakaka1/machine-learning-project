"""Apply the acceptance rule of `docs/revision_protocolo.md` to every family of experiments.

It reads the tables that the other modules wrote and trains nothing.
Run it with `uv run python -m experiments.review.decide`.
"""

import pandas as pd

from experiments.review import harness

_FAMILIES: tuple[tuple[str, str], ...] = (
    ("treatments", "as_is"),
    ("tiers", "all_19"),
    ("ablation", "all_19"),
    ("engineered", "all_19"),
    ("parsimony", "forest_19"),
)
_COLUMNS = ("roc_auc", "roc_auc_no_deposit")


def _accepted(row: pd.Series, gaps_q4: pd.Series) -> tuple[bool, str]:
    reasons = []
    gains = {metric: row[f"{metric}_gap"] for metric in _COLUMNS}
    best = max(gains, key=lambda metric: gains[metric])
    if gains[best] < harness.MIN_GAIN:
        reasons.append("ganancia por debajo de +0.002")
    if not (row[f"{best}_low"] > 0 and row[f"{best}_p_holm"] < 0.05):
        reasons.append("el intervalo o Holm no excluyen el cero")
    other = next(metric for metric in _COLUMNS if metric != best)
    if row[f"{other}_high"] < 0:
        reasons.append("empeora la otra métrica")
    if row["log_loss_low"] > 0:
        reasons.append("empeora el log-loss")
    if gaps_q4[f"{best}_gap"] <= 0:
        reasons.append("cambia de signo en Q4")
    return not reasons, "; ".join(reasons) if reasons else "cumple todo"


def main() -> None:
    """Write one row per comparison, with the verdict of the rule."""
    log = []
    for family, _reference in _FAMILIES:
        monthly = pd.read_csv(harness.OUTPUT_DIR / f"{family}_m12.csv")
        quarterly = pd.read_csv(harness.OUTPUT_DIR / f"{family}_q4.csv").set_index("variant")
        for _, row in monthly[monthly["roc_auc_gap"].notna()].iterrows():
            accepted, reason = _accepted(row, quarterly.loc[row["variant"]])
            log.append(
                {
                    "family": family,
                    "variant": row["variant"],
                    "m12_roc_auc": row["roc_auc"],
                    "m12_gap": row["roc_auc_gap"],
                    "m12_gap_low": row["roc_auc_low"],
                    "m12_gap_high": row["roc_auc_high"],
                    "m12_p_holm": row["roc_auc_p_holm"],
                    "m12_no_deposit": row["roc_auc_no_deposit"],
                    "m12_no_deposit_gap": row["roc_auc_no_deposit_gap"],
                    "m12_no_deposit_low": row["roc_auc_no_deposit_low"],
                    "m12_no_deposit_high": row["roc_auc_no_deposit_high"],
                    "m12_no_deposit_p_holm": row["roc_auc_no_deposit_p_holm"],
                    "m12_log_loss_gap": row["log_loss_gap"],
                    "q4_gap": quarterly.loc[row["variant"], "roc_auc_gap"],
                    "q4_no_deposit_gap": quarterly.loc[row["variant"], "roc_auc_no_deposit_gap"],
                    "accepted": accepted,
                    "reason": reason,
                }
            )
    table = pd.DataFrame(log)
    harness.save("decision_log", table)
    pd.set_option("display.width", 240, "display.max_columns", 40, "display.max_rows", 100)
    print(
        table[
            [
                "family",
                "variant",
                "m12_gap",
                "m12_no_deposit_gap",
                "m12_no_deposit_p_holm",
                "q4_gap",
                "accepted",
                "reason",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
