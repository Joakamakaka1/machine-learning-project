"""Post-hoc reading of the stored predictions on 2017. It is not a new evaluation.

It scores nothing: the probabilities come from `outputs/final_evaluation/predictions.csv.gz`. The
only model trained here is a drift classifier, which never sees the target. The scope is fixed in
the amendment of `docs/revision_protocolo.md`, and no decision comes out of it.

Run it with `uv run python -m experiments.review.posthoc_2017`.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from experiments import common
from experiments.review import harness
from hotel_cancellation import config
from hotel_cancellation.data import data_loader, features, schema

_REPLICATES = 2000
_THRESHOLD = 0.25
_MIN_EVENT = 10


def load_test_with_predictions() -> pd.DataFrame:
    """Align the stored probabilities with the clean rows of 2017, and check that they match."""
    raw = data_loader.load_raw()
    test = data_loader.split_temporal(data_loader.clean(raw)).test
    stored = pd.read_csv(config.get_settings().final_evaluation_dir / "predictions.csv.gz")
    same_rows = (
        len(stored) == len(test)
        and (stored[schema.TARGET].to_numpy() == test[schema.TARGET].to_numpy()).all()
        and (stored["hotel"].to_numpy() == test["hotel"].to_numpy()).all()
        and (stored["deposit_type"].to_numpy() == test["deposit_type"].to_numpy()).all()
        and (
            pd.to_datetime(stored[schema.ARRIVAL_DATE]).to_numpy()
            == test[schema.ARRIVAL_DATE].to_numpy()
        ).all()
    )
    if not same_rows:
        raise ValueError("The stored predictions are not aligned with the rows of 2017.")
    return harness.add_derived(test).assign(score=stored["champion"].to_numpy())


def intervals(rows: pd.DataFrame) -> pd.DataFrame:
    """Bootstrap the published figure with blocks that reflect different dependencies."""
    actual, scores = rows[schema.TARGET].to_numpy(), rows["score"].to_numpy()
    blockings = {
        "hotel_arrival_day": rows["hotel"] + "|" + rows[schema.ARRIVAL_DATE].astype("str"),
        "arrival_week": rows[schema.ARRIVAL_DATE].dt.to_period("W").astype("str"),
        "booking_batch": rows["batch_id"],
        "hotel_status_day_agent": rows["hotel"]
        + "|"
        + rows["status_date"].astype("str")
        + "|"
        + rows["agent_key"],
        "agent_booking_month": rows["agent_key"]
        + "|"
        + rows["booking_date"].dt.to_period("M").astype("str"),
    }
    results = []
    for name, blocks in blockings.items():
        codes, names = pd.factorize(blocks)
        rng = np.random.default_rng(0)
        samples = []
        for _ in range(_REPLICATES):
            weights = np.bincount(rng.integers(0, names.size, names.size), minlength=names.size)[
                codes
            ]
            samples.append(roc_auc_score(actual, scores, sample_weight=weights))
        low, high = np.quantile(samples, [0.025, 0.975])
        results.append(
            {
                "blocks": name,
                "block_count": names.size,
                "roc_auc": float(roc_auc_score(actual, scores)),
                "ci_low": float(low),
                "ci_high": float(high),
                "width": float(high - low),
            }
        )
        print(results[-1], flush=True)
    return pd.DataFrame(results)


def by_booking_date(rows: pd.DataFrame) -> pd.DataFrame:
    """Measure the rows by when they were confirmed, against what the model had learned by then."""
    booked = rows["booking_date"]
    subsets = {
        "all": pd.Series(True, index=rows.index),
        "booked_from_2017_01_01": booked >= "2017-01-01",
        "booked_from_2016_10_01": booked >= "2016-10-01",
        "booked_before_2016_10_01": booked < "2016-10-01",
    }
    results = []
    for name, mask in subsets.items():
        group = rows[mask]
        results.append(
            {
                "subset": name,
                "non_refund_share": float((group["deposit_type"] == "Non Refund").mean()),
                "longest_lead_time": int(group["lead_time"].max()),
                **harness.score_window(group, group["score"].to_numpy()),
            }
        )
    return pd.DataFrame(results)


def positive_kinds(rows: pd.DataFrame) -> pd.DataFrame:
    """Rank each kind of cancellation against the stays, as `errors.py` does on 2016."""
    canceled = rows["reservation_status"] == "Canceled"
    days_before = (rows[schema.ARRIVAL_DATE] - rows["status_date"]).dt.days
    no_deposit = rows["deposit_type"] == "No Deposit"
    event_size = harness.cancellation_event_size(rows)
    kinds = {
        "no_show": rows["reservation_status"] == "No-Show",
        "canceled_no_deposit": canceled & no_deposit,
        "canceled_no_deposit_alone": canceled & no_deposit & (event_size < _MIN_EVENT),
        "canceled_no_deposit_in_event_10_or_more": canceled
        & no_deposit
        & (event_size >= _MIN_EVENT),
        "canceled_within_7_days_of_arrival": canceled & (days_before <= 7),
        "canceled_8_to_30_days_before": canceled & (days_before > 7) & (days_before <= 30),
        "canceled_31_or_more_days_before": canceled & (days_before > 30),
    }
    stays = rows[rows[schema.TARGET] == 0]
    results = []
    for kind, mask in kinds.items():
        positives = rows[mask]
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


def flagged_share(rows: pd.DataFrame) -> pd.DataFrame:
    """Give the share of bookings at or above the served threshold, in total and per month."""
    flagged = rows["score"] >= _THRESHOLD
    month = rows[schema.ARRIVAL_DATE].dt.to_period("M").astype("str")
    table = pd.DataFrame({"flagged": flagged, "y": rows[schema.TARGET], "month": month})
    monthly = table.groupby("month").agg(
        rows=("y", "size"), prevalence=("y", "mean"), flagged=("flagged", "mean")
    )
    total = pd.DataFrame(
        {
            "rows": [len(table)],
            "prevalence": [table["y"].mean()],
            "flagged": [table["flagged"].mean()],
        },
        index=["total"],
    )
    return pd.concat([total, monthly]).reset_index(names="period")


def _drift_auc(first: pd.DataFrame, second: pd.DataFrame, drop: tuple[str, ...] = ()) -> float:
    """Train on half of each period to tell them apart, and score the other half by batch."""
    numeric = [column for column in schema.NUMERIC_FEATURES if column not in drop]
    categorical = [column for column in schema.CATEGORICAL_FEATURES if column not in drop]
    both = pd.concat([first.assign(period=0), second.assign(period=1)], ignore_index=True)
    rng = np.random.default_rng(common.SEED)
    keys = both["hotel"] + both["booking_date"].astype("str") + both["agent_key"]
    codes, names = pd.factorize(keys)
    in_fit = rng.random(names.size)[codes] < 0.5
    pipeline = features.build_pipeline(
        features.build_preprocessor(
            features.Family.UNSCALED, numeric=numeric, categorical=categorical
        ),
        common.reference_xgboost(),
    )
    frame = data_loader.coerce_features(both)
    pipeline.fit(frame[in_fit], both.loc[in_fit, "period"])
    return float(
        roc_auc_score(both.loc[~in_fit, "period"], pipeline.predict_proba(frame[~in_fit])[:, 1])
    )


def drift(rows: pd.DataFrame) -> pd.DataFrame:
    """Measure how separable the periods are, with and without the two variables most exposed."""
    development = harness.load_rows()
    dates = development[schema.ARRIVAL_DATE]
    base = development[dates < "2016-10-01"]
    control_first = development[dates < "2016-07-01"]
    control_second = development[dates >= "2016-07-01"]
    sets = {
        "all_19": (),
        "without_agent": ("agent",),
        "without_lead_time": ("lead_time",),
        "without_agent_and_adr": ("agent", "adr"),
    }
    results = []
    for name, drop in sets.items():
        results.append(
            {
                "inputs": name,
                "base_training_vs_2017": _drift_auc(base, rows, drop),
                "control_until_2016_06_vs_2016_second_half": _drift_auc(
                    control_first, control_second, drop
                ),
            }
        )
    return pd.DataFrame(results)


def main() -> None:
    """Write the tables of the post-hoc reading."""
    rows = load_test_with_predictions()
    pd.set_option("display.width", 220, "display.max_columns", 30)
    for name, step in (
        ("posthoc_by_booking_date", by_booking_date),
        ("posthoc_positive_kinds", positive_kinds),
        ("posthoc_flagged", flagged_share),
        ("posthoc_drift", drift),
        ("posthoc_intervals", intervals),
    ):
        table = step(rows)
        harness.save(name, table)
        print(f"\n== {name} ==\n{table.round(4).to_string(index=False)}\n", flush=True)


if __name__ == "__main__":
    main()
