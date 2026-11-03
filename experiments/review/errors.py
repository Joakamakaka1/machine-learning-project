"""Where and why the champion's configuration fails, on pooled predictions of the four windows.

Each row is predicted by the model trained before its window. The base model is not calibrated, so
this looks at ranking. Run it with `uv run python -m experiments.review.errors`.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from experiments import common
from experiments.review import harness
from hotel_cancellation import data_loader, schema

_MIN_ROWS = 100


def pooled_predictions(rows: pd.DataFrame) -> pd.DataFrame:
    """Predict each row of 2016 with the champion's configuration trained before its window."""
    dates = rows[schema.ARRIVAL_DATE]
    parts = []
    for window in harness.Q4:
        train = rows[dates < window.train_end]
        valid = rows[(dates >= window.start) & (dates < window.end)]
        pipeline = common.tuned_pipeline("random_forest").set_params(model__n_jobs=-1)
        pipeline.fit(data_loader.coerce_features(train), train[schema.TARGET])
        seen_agents = train["agent_key"].value_counts()
        parts.append(
            valid.assign(
                score=pipeline.predict_proba(data_loader.coerce_features(valid))[:, 1],
                window=window.name,
                agent_rows_in_train=valid["agent_key"].map(seen_agents).fillna(0).astype(int),
            )
        )
    return pd.concat(parts)


def _segment_columns(pooled: pd.DataFrame) -> pd.DataFrame:
    return pooled.assign(
        lead_band=pd.cut(pooled["lead_time"], [-1, 7, 30, 90, 180, 365, 10_000]).astype("str"),
        nights_band=pd.cut(pooled["nights"], [0, 1, 3, 7, 14, 100]).astype("str"),
        batch_band=pd.cut(pooled["batch_size"], [0, 1, 4, 9, 29, 10_000]).astype("str"),
        agent_band=pd.cut(pooled["agent_rows_in_train"], [-1, 0, 49, 499, 10**7]).astype("str"),
        any_previous_cancellation=pooled["previous_cancellations"] > 0,
        event_size=harness.cancellation_event_size(pooled),
    )


def by_segment(pooled: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Compute ranking and error figures per value of each segment column."""
    results = []
    for column in columns:
        for value, group in pooled.groupby(column, observed=True):
            if len(group) < _MIN_ROWS:
                continue
            actual, scores = group[schema.TARGET].to_numpy(), group["score"].to_numpy()
            predicted = scores >= 0.5
            results.append(
                {
                    "segment": column,
                    "value": str(value),
                    "rows": len(group),
                    "prevalence": float(actual.mean()),
                    "mean_score": float(scores.mean()),
                    "roc_auc": harness.safe_auc(actual, scores),
                    "brier": float(brier_score_loss(actual, scores)),
                    "missed_share_of_all_misses": float(
                        ((~predicted) & (actual == 1)).sum()
                        / max(((pooled["score"] < 0.5) & (pooled[schema.TARGET] == 1)).sum(), 1)
                    ),
                    "false_alarm_share_of_all": float(
                        (predicted & (actual == 0)).sum()
                        / max(((pooled["score"] >= 0.5) & (pooled[schema.TARGET] == 0)).sum(), 1)
                    ),
                }
            )
    return pd.DataFrame(results)


def positives_by_kind(pooled: pd.DataFrame) -> pd.DataFrame:
    """Rank each kind of cancellation against the stays: which ones can the model see coming?"""
    stays = pooled[pooled[schema.TARGET] == 0]
    canceled = pooled["reservation_status"] == "Canceled"
    days_before = (pooled[schema.ARRIVAL_DATE] - pooled["status_date"]).dt.days
    life = 1 - days_before / pooled["lead_time"].where(pooled["lead_time"] > 0)
    kinds = {
        "no_show": pooled["reservation_status"] == "No-Show",
        "canceled_non_refund": canceled & (pooled["deposit_type"] == "Non Refund"),
        "canceled_no_deposit": canceled & (pooled["deposit_type"] == "No Deposit"),
        "canceled_no_deposit_alone": canceled
        & (pooled["deposit_type"] == "No Deposit")
        & (pooled["event_size"] < 10),
        "canceled_no_deposit_in_event_10_or_more": canceled
        & (pooled["deposit_type"] == "No Deposit")
        & (pooled["event_size"] >= 10),
        "canceled_within_7_days_of_arrival": canceled & (days_before <= 7),
        "canceled_8_to_30_days_before": canceled & (days_before > 7) & (days_before <= 30),
        "canceled_31_or_more_days_before": canceled & (days_before > 30),
        "canceled_first_5_percent_of_life": canceled & (life <= 0.05),
        "canceled_last_25_percent_of_life": canceled & (life > 0.75),
    }
    results = []
    for kind, mask in kinds.items():
        positives = pooled[mask]
        actual = np.r_[np.ones(len(positives)), np.zeros(len(stays))]
        scores = np.r_[positives["score"].to_numpy(), stays["score"].to_numpy()]
        results.append(
            {
                "kind": kind,
                "positives": len(positives),
                "auc_against_stays": harness.safe_auc(actual, scores),
                "mean_score": float(positives["score"].mean()),
            }
        )
    return pd.DataFrame(results)


def main() -> None:
    """Write the error tables."""
    rows = harness.load_rows()
    pd.set_option("display.width", 240, "display.max_columns", 30, "display.max_rows", 200)
    pooled = _segment_columns(pooled_predictions(rows))
    segments = by_segment(
        pooled,
        [
            "hotel",
            "deposit_type",
            "market_segment",
            "distribution_channel",
            "customer_type",
            "lead_band",
            "nights_band",
            "batch_band",
            "agent_band",
            "any_previous_cancellation",
            "is_repeated_guest",
            "window",
        ],
    )
    kinds = positives_by_kind(pooled)
    harness.save("errors_segments", segments)
    harness.save("errors_positive_kinds", kinds)
    print(segments.round(3).to_string(index=False))
    print(kinds.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
