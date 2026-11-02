"""The experiments on synthetic rows, with a small reference model. They need the train extra."""

from typing import Any

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from experiments import common, decisions, encodings
from hotel_cancellation import config, data_loader
from tests.conftest import synthetic_raw


@pytest.fixture(autouse=True)
def small_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    def small(**overrides: Any) -> XGBClassifier:
        return XGBClassifier(
            **{
                "n_estimators": 10,
                "max_depth": 2,
                "n_jobs": 1,
                "random_state": 0,
                **overrides,
            }
        )

    monkeypatch.setattr(common, "reference_xgboost", small)


def _constant(_: pd.DataFrame, valid: pd.DataFrame) -> np.ndarray:
    return np.full(len(valid), 0.5)


def _lead_time(_: pd.DataFrame, valid: pd.DataFrame) -> np.ndarray:
    return np.asarray(valid["lead_time"].to_numpy() / 300)


def test_experiments_refuse_rows_of_the_test_period() -> None:
    cleaned = data_loader.clean(synthetic_raw(rows_per_month=10))
    with pytest.raises(ValueError, match="2017-01-01"):
        common.compare("x", cleaned, [common.Variant("a", _constant)])


def test_compare_gives_a_row_per_variant_and_window(synthetic_development: pd.DataFrame) -> None:
    result = common.compare(
        "x",
        synthetic_development,
        [common.Variant("constant", _constant), common.Variant("lead_time", _lead_time)],
    )
    assert len(result.by_window) == 2 * len(config.EXPANDING_WINDOWS)
    summary: Any = result.summary.set_index("variant")
    assert summary.loc["constant", "roc_auc"] == pytest.approx(0.5)
    assert summary.loc["constant", "gap"] == 0
    assert summary.loc["lead_time", "gap"] == pytest.approx(
        summary.loc["lead_time", "roc_auc"] - 0.5
    )
    assert summary.loc["lead_time", "paired_se"] > 0


def test_each_window_trains_on_earlier_arrivals_only(synthetic_development: pd.DataFrame) -> None:
    def checked(train: pd.DataFrame, valid: pd.DataFrame) -> np.ndarray:
        assert train["arrival_date"].max() < valid["arrival_date"].min()
        return _constant(train, valid)

    common.compare("x", synthetic_development, [common.Variant("checked", checked)])


def test_special_requests_scenario_only_adds_that_variable(
    synthetic_development: pd.DataFrame,
) -> None:
    result = decisions.special_requests(synthetic_development, tuned=None)
    summary: Any = result.summary.set_index("variant")
    # On synthetic rows the variable copies the outcome, so adding it must be obvious.
    assert summary.loc["xgboost_plus_special_requests", "roc_auc"] > 0.99
    assert summary.loc["xgboost_19_variables", "roc_auc"] < 0.95


def test_calendar_compares_the_final_variables_with_the_calendar(
    synthetic_development: pd.DataFrame,
) -> None:
    result = decisions.calendar(synthetic_development)
    assert result.summary["variant"].iloc[0] == "19_variables"
    assert len(result.summary) == 4


def test_decisions_run_on_synthetic_rows(synthetic_development: pd.DataFrame) -> None:
    for experiment in (decisions.duplicates, decisions.per_hotel, decisions.tree_depth):
        result = experiment(synthetic_development)
        assert result.summary["roc_auc"].between(0.5, 1).all()


def test_every_encoding_runs_on_synthetic_rows(synthetic_development: pd.DataFrame) -> None:
    summary = encodings.encodings(synthetic_development).summary
    assert len(summary) == 5
    assert summary["roc_auc"].between(0.5, 1).all()


@pytest.mark.slow
def test_every_balancing_method_runs_on_synthetic_rows(
    synthetic_development: pd.DataFrame,
) -> None:
    pytest.importorskip("imblearn")
    from experiments import balancing

    summary: Any = balancing.balancing(synthetic_development).summary.set_index("variant")
    assert summary["roc_auc"].between(0.5, 1).all()
    # Balancing teaches a prevalence of one half, so the probabilities must get worse.
    assert summary.loc["logistic_undersampling", "brier"] > summary.loc["logistic_none", "brier"]


@pytest.mark.slow
def test_embedding_mlp_learns_the_signal(synthetic_development: pd.DataFrame) -> None:
    pytest.importorskip("keras")
    from experiments import embeddings

    dates = synthetic_development["arrival_date"]
    train = synthetic_development[dates < pd.Timestamp("2016-07-01")]
    valid = synthetic_development[dates >= pd.Timestamp("2016-07-01")]
    probabilities: Any = embeddings.embedding_mlp(hidden_layers=(16,), batch_size=32, epochs=8)(
        train, valid
    )
    assert probabilities.shape == (len(valid),)
    frame: Any = valid.assign(probability=probabilities)
    higher = frame[frame["deposit_type"] == "Non Refund"]["probability"].mean()
    lower = frame[frame["deposit_type"] == "No Deposit"]["probability"].mean()
    assert higher > lower
