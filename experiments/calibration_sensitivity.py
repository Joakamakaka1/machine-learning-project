"""How much does the calibration slope depend on the clip of extreme probabilities?

An isotonic calibrator gives some rows a probability of exactly 0 or 1. Each model repeats the
calibration comparison of the protocol: base before 2016T3, calibrator on 2016T3 and measure on
2016T4, with the method it chose. The served models stay untouched.

Run it with `uv run python -m experiments.calibration_sensitivity`.
"""

import json
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from experiments import common
from hotel_cancellation import config, evaluator, model_trainer, models
from hotel_cancellation.data import data_loader, schema


def sensitivity(
    development: pd.DataFrame,
    catalog: Sequence[models.ModelSpec],
    params: Mapping[str, Mapping[str, Any]],
    methods: Mapping[str, str],
) -> pd.DataFrame:
    """Measure the slope of each model on 2016T4 with the three clips.

    Args:
        development: Development rows.
        catalog: Models to measure.
        params: Hyperparameters of each model.
        methods: Calibration method of each model.

    Returns:
        A row per model.

    Raises:
        ValueError: If a row belongs to the test period.
    """
    dates = development[schema.ARRIVAL_DATE]
    if (dates >= config.TEST_START).any():
        raise ValueError(f"Experiments only accept arrivals before {config.TEST_START.date()}.")
    cut = config.CALIBRATION_SELECTION
    frame = data_loader.coerce_features(development)
    target = development[schema.TARGET]
    base_rows = dates < cut.base_end
    calibration_rows = (dates >= cut.base_end) & (dates < cut.calibration_end)
    measured_rows = dates >= cut.calibration_end

    rows = []
    for spec in catalog:
        model = spec.pipeline(common.SEED).set_params(**params[spec.name])
        model.fit(frame[base_rows], target[base_rows])
        method = methods[spec.name]
        if method != model_trainer.NO_CALIBRATION:
            model = CalibratedClassifierCV(FrozenEstimator(model), method=method).fit(
                frame[calibration_rows], target[calibration_rows]
            )
        probabilities = model.predict_proba(frame[measured_rows])[:, 1]
        rows.append(
            {
                "model": spec.name,
                "calibration": method,
                **evaluator.calibration_diagnostics(target[measured_rows], probabilities),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Measure the real models and store the table."""
    settings = config.get_settings()
    search = json.loads(settings.best_params_path.read_text("utf-8"))["models"]
    calibration = json.loads((settings.validation_dir / "calibration.json").read_text("utf-8"))
    table = sensitivity(
        common.load_development(),
        models.model_catalog("full"),
        {name: record["params"] for name, record in search.items()},
        {name: record["method"] for name, record in calibration.items()},
    )
    settings.experiments_dir.mkdir(parents=True, exist_ok=True)
    table.round(5).to_csv(
        settings.experiments_dir / "calibration_sensitivity.csv", index=False, lineterminator="\n"
    )
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
