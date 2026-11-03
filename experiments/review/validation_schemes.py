"""Diagnose the validation scheme itself, before comparing anything with it.

Each table answers one question about the windows of the protocol: do booking batches leak across
them, do rows booked before the cut flatter them, how fast does a model go stale, how much does a
random split inflate the figures and how wide should the intervals be.

Run it with `uv run python -m experiments.review.validation_schemes`.
"""

import sys
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

from experiments import common
from experiments.review import harness
from hotel_cancellation import data_loader, evaluator, schema

_MIN_COHORT_TRAIN_START = pd.Timestamp("2015-10-01")
_BOOTSTRAP_REPLICATES = 1000
_MONTH = pd.offsets.MonthBegin(1)


def _make_models() -> dict[str, Callable[[], Any]]:
    return {
        "xgboost": common.reference_xgboost,
        "random_forest": lambda: common.tuned_pipeline("random_forest").set_params(
            model__n_jobs=-1
        ),
    }


def _fit_scores(make_model: Callable[[], Any], train: pd.DataFrame, valid: pd.DataFrame) -> Any:
    """Fit a model, which is an estimator or a whole pipeline, and score the validation rows."""
    model = make_model()
    if hasattr(model, "steps"):
        model.fit(data_loader.coerce_features(train), train[schema.TARGET])
        return model.predict_proba(data_loader.coerce_features(valid))[:, 1]
    pipeline = common.tabular_predictor(lambda: model)
    return pipeline(train, valid)


def purged_folds(rows: pd.DataFrame) -> pd.DataFrame:
    """Remove from training the rows of a batch that is also in validation, and compare."""
    results = []
    loose_key = rows["hotel"] + "|" + rows["booking_date"].astype("str") + "|" + rows["agent_key"]
    dates = rows[schema.ARRIVAL_DATE]
    for window in harness.Q4:
        train = rows[dates < window.train_end]
        valid = rows[(dates >= window.start) & (dates < window.end)]
        strict = ~train["batch_id"].isin(set(valid["batch_id"]))
        loose = ~loose_key[train.index].isin(set(loose_key[valid.index]))
        for name, make_model in _make_models().items():
            schemes = {
                "all_train": train,
                "purged_batch": train[strict],
                "purged_booking_day_agent": train[loose],
            }
            for scheme, subset in schemes.items():
                scores = harness.score_window(valid, _fit_scores(make_model, subset, valid))
                results.append(
                    {
                        "model": name,
                        "scheme": scheme,
                        "window": window.name,
                        "train_rows": len(subset),
                        **scores,
                    }
                )
                print(
                    f"purge | {window.name} | {name} | {scheme}: {scores['roc_auc']:.4f}",
                    flush=True,
                )
    return pd.DataFrame(results)


def booked_after_cut(rows: pd.DataFrame) -> pd.DataFrame:
    """Measure the protocol's model on rows booked before and after its cut, by lead time band."""
    results = []
    dates = rows[schema.ARRIVAL_DATE]
    for window in harness.Q4:
        train = rows[dates < window.train_end]
        valid = rows[(dates >= window.start) & (dates < window.end)]
        scores = np.asarray(_fit_scores(common.reference_xgboost, train, valid))
        after = (valid["booking_date"] >= window.start).to_numpy()
        short_lead = (valid["lead_time"] < 30).to_numpy()
        no_deposit = (valid["deposit_type"] == "No Deposit").to_numpy()
        subsets = {
            "all": np.ones(len(valid), dtype=bool),
            "booked_after_cut": after,
            "booked_before_cut": ~after,
            "lead_under_30_after": after & short_lead,
            "lead_under_30_before": ~after & short_lead,
            "no_deposit_after": after & no_deposit,
            "no_deposit_before": ~after & no_deposit,
        }
        actual = valid[schema.TARGET].to_numpy()
        for subset, mask in subsets.items():
            if mask.sum() < 50:
                continue
            results.append(
                {
                    "window": window.name,
                    "subset": subset,
                    "rows": int(mask.sum()),
                    "prevalence": float(actual[mask].mean()),
                    "roc_auc": harness.safe_auc(actual[mask], scores[mask]),
                }
            )
    return pd.DataFrame(results)


def strict_point_in_time(rows: pd.DataFrame) -> pd.DataFrame:
    """Compare the protocol's training set with the one known when each row was booked.

    Each booking month gets a model trained on arrivals before that month, which is what a monthly
    retrain would have had at that moment. Only rows booked from October 2015 on have enough past.
    """
    results = []
    dates = rows[schema.ARRIVAL_DATE]
    for window in harness.Q4:
        train = rows[dates < window.train_end]
        valid = rows[(dates >= window.start) & (dates < window.end)]
        valid = valid[valid["booking_date"] >= _MIN_COHORT_TRAIN_START]
        protocol = np.asarray(_fit_scores(common.reference_xgboost, train, valid))
        strict = np.zeros(len(valid))
        cohorts = valid["booking_date"].dt.to_period("M").dt.to_timestamp()
        for cohort in sorted(cohorts.unique()):
            cut = min(pd.Timestamp(cohort), window.start)
            members = (cohorts == cohort).to_numpy()
            past = rows[dates < cut]
            strict[members] = np.asarray(
                _fit_scores(common.reference_xgboost, past, valid[members])
            )
        actual = valid[schema.TARGET].to_numpy()
        before = (valid["booking_date"] < window.start).to_numpy()
        for subset, mask in (
            ("all", np.ones(len(valid), dtype=bool)),
            ("booked_before_cut", before),
        ):
            results.append(
                {
                    "window": window.name,
                    "subset": subset,
                    "rows": int(mask.sum()),
                    "protocol_roc_auc": harness.safe_auc(actual[mask], protocol[mask]),
                    "strict_roc_auc": harness.safe_auc(actual[mask], strict[mask]),
                }
            )
        print(f"strict | {window.name}: {results[-2]}", flush=True)
    return pd.DataFrame(results)


def horizon_decay(rows: pd.DataFrame) -> pd.DataFrame:
    """Score the same validation quarter with models whose training ends further back."""
    results = []
    dates = rows[schema.ARRIVAL_DATE]
    for window in harness.Q4:
        valid = rows[(dates >= window.start) & (dates < window.end)]
        for gap_months in (0, 3, 6, 9):
            end = window.start - pd.DateOffset(months=gap_months)
            train = rows[dates < end]
            if len(train) < 5000:
                continue
            for name, make_model in _make_models().items():
                scores = harness.score_window(valid, _fit_scores(make_model, train, valid))
                results.append(
                    {
                        "model": name,
                        "window": window.name,
                        "gap_months": gap_months,
                        "train_rows": len(train),
                        **scores,
                    }
                )
                print(
                    f"horizon | {window.name} | {gap_months} | {name}: {scores['roc_auc']:.4f}",
                    flush=True,
                )
    return pd.DataFrame(results)


def random_split_contrast(rows: pd.DataFrame) -> pd.DataFrame:
    """Measure, inside development, how much random and grouped folds inflate the figures."""
    results = []
    actual = rows[schema.TARGET].to_numpy()
    splitters = {
        "random_stratified": StratifiedKFold(5, shuffle=True, random_state=common.SEED).split(
            rows, actual
        ),
        "grouped_by_batch": GroupKFold(5).split(rows, actual, rows["batch_id"]),
    }
    for scheme, folds in splitters.items():
        for fold, (train_index, valid_index) in enumerate(folds):
            train, valid = rows.iloc[train_index], rows.iloc[valid_index]
            scores = harness.score_window(
                valid, _fit_scores(common.reference_xgboost, train, valid)
            )
            results.append({"scheme": scheme, "window": f"fold{fold}", **scores})
            print(f"random | {scheme} | {fold}: {scores['roc_auc']:.4f}", flush=True)
    return pd.DataFrame(results)


def _interval(
    actual: Any, scores: Any, blocks: pd.Series, *, replicates: int = _BOOTSTRAP_REPLICATES
) -> tuple[float, float]:
    codes, names = pd.factorize(blocks)
    rng = np.random.default_rng(0)
    samples = []
    for _ in range(replicates):
        counts = np.bincount(rng.integers(0, names.size, names.size), minlength=names.size)
        weights = counts[codes]
        samples.append(roc_auc_score(actual, scores, sample_weight=weights))
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def interval_width(rows: pd.DataFrame) -> pd.DataFrame:
    """Compare interval widths of block bootstraps on pooled out-of-window predictions."""
    dates = rows[schema.ARRIVAL_DATE]
    parts = []
    for window in harness.Q4:
        train = rows[dates < window.train_end]
        valid = rows[(dates >= window.start) & (dates < window.end)]
        parts.append(valid.assign(score=_fit_scores(_make_models()["random_forest"], train, valid)))
    pooled = pd.concat(parts)
    hotel_day = pooled["hotel"] + "|" + pooled[schema.ARRIVAL_DATE].astype("str")
    week = pooled[schema.ARRIVAL_DATE].dt.to_period("W").astype("str")
    event = pooled["hotel"] + "|" + pooled["status_date"].astype("str") + "|" + pooled["agent_key"]
    agent_month = pooled["agent_key"] + "|" + pooled["booking_date"].dt.to_period("M").astype("str")
    blockings = {
        "row": pd.Series(np.arange(len(pooled)), index=pooled.index),
        "hotel_arrival_day": hotel_day,
        "arrival_week": week,
        "booking_batch": pooled["batch_id"],
        "hotel_status_day_agent": event,
        "agent_booking_month": agent_month,
        "agent": pooled["agent_key"],
    }
    actual, scores = pooled[schema.TARGET].to_numpy(), pooled["score"].to_numpy()
    results = []
    for name, blocks in blockings.items():
        low, high = _interval(actual, scores, blocks)
        results.append(
            {
                "blocks": name,
                "block_count": int(blocks.nunique()),
                "roc_auc": float(roc_auc_score(actual, scores)),
                "ci_low": low,
                "ci_high": high,
                "width": high - low,
            }
        )
        print(results[-1], flush=True)
    return pd.DataFrame(results)


def unique_definitions(rows: pd.DataFrame) -> pd.DataFrame:
    """Compare the published definition of unique rows with the one that ignores the outcome."""
    dates = rows[schema.ARRIVAL_DATE]
    results = []
    for window in harness.Q4:
        train = rows[dates < window.train_end]
        valid = rows[(dates >= window.start) & (dates < window.end)]
        scores = _fit_scores(_make_models()["random_forest"], train, valid)
        published = evaluator.validation_metrics(
            valid[schema.TARGET], scores, data_loader.coerce_features(valid)
        )
        review = harness.score_window(valid, scores)
        results.append(
            {
                "window": window.name,
                "roc_auc": review["roc_auc"],
                "unique_with_target_in_key": published["roc_auc_unique"],
                "distinct_inputs_only": review["roc_auc_distinct"],
            }
        )
    return pd.DataFrame(results)


def purge_control(rows: pd.DataFrame) -> pd.DataFrame:
    """Separate leakage from data loss: remove as many random batches as the purge removes.

    A purge also shrinks training. Dropping unrelated batches of the same total size gives the loss
    that comes from having fewer rows, and the purge only signals leakage beyond that.
    """
    results = []
    dates = rows[schema.ARRIVAL_DATE]
    for window in harness.Q4:
        train = rows[dates < window.train_end]
        valid = rows[(dates >= window.start) & (dates < window.end)]
        purged = ~train["batch_id"].isin(set(valid["batch_id"]))
        removed = int((~purged).sum())
        for name, make_model in _make_models().items():
            schemes: dict[str, pd.DataFrame] = {"all_train": train, "purged_batch": train[purged]}
            for repetition in range(3):
                rng = np.random.default_rng(common.SEED + repetition)
                unrelated = train.loc[purged, "batch_id"].unique()
                dropped: set[int] = set()
                total = 0
                sizes = train.groupby("batch_id").size()
                for batch in rng.permutation(unrelated):
                    if total >= removed:
                        break
                    dropped.add(int(batch))
                    total += int(sizes[batch])
                schemes[f"random_control_{repetition}"] = train[~train["batch_id"].isin(dropped)]
            for scheme, subset in schemes.items():
                scores = harness.score_window(valid, _fit_scores(make_model, subset, valid))
                results.append(
                    {
                        "model": name,
                        "scheme": scheme,
                        "window": window.name,
                        "train_rows": len(subset),
                        **scores,
                    }
                )
                print(
                    f"control | {window.name} | {name} | {scheme}: {scores['roc_auc']:.4f}",
                    flush=True,
                )
    return pd.DataFrame(results)


def prospective_cohorts(rows: pd.DataFrame) -> pd.DataFrame:
    """Measure what a model deployed at a date sees: bookings confirmed after it, at any arrival.

    Long lead times are cut off because arrivals after 2016 belong to the test period, so the later
    origins see shorter leads. The table gives the longest lead time each origin can reach.
    """
    results = []
    dates = rows[schema.ARRIVAL_DATE]
    for window in harness.Q4:
        train = rows[dates < window.train_end]
        cohort = rows[(rows["booking_date"] >= window.start) & (rows["booking_date"] < window.end)]
        for name, make_model in _make_models().items():
            scores = harness.score_window(cohort, _fit_scores(make_model, train, cohort))
            results.append(
                {
                    "model": name,
                    "origin": window.name,
                    "longest_lead_time": int(cohort["lead_time"].max()),
                    "non_refund_share": float((cohort["deposit_type"] == "Non Refund").mean()),
                    **scores,
                }
            )
            print(f"cohort | {window.name} | {name}: {scores['roc_auc']:.4f}", flush=True)
    return pd.DataFrame(results)


def main() -> None:
    """Write the tables named on the command line, or all of them, to `outputs/review/`."""
    rows = harness.load_rows()
    pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_rows", 200)
    steps: tuple[tuple[str, Callable[[pd.DataFrame], pd.DataFrame]], ...] = (
        ("validation_unique_definitions", unique_definitions),
        ("validation_purged", purged_folds),
        ("validation_purge_control", purge_control),
        ("validation_booked_after_cut", booked_after_cut),
        ("validation_prospective_cohorts", prospective_cohorts),
        ("validation_strict_point_in_time", strict_point_in_time),
        ("validation_horizon", horizon_decay),
        ("validation_random_contrast", random_split_contrast),
        ("validation_interval_width", interval_width),
    )
    chosen = set(sys.argv[1:])
    for name, step in steps:
        if chosen and name not in chosen:
            continue
        table = step(rows)
        harness.save(name, table)
        print(f"\n== {name} ==\n{table.round(4).to_string(index=False)}\n", flush=True)


if __name__ == "__main__":
    main()
