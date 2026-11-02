"""Run every design experiment on arrivals before 2017 and store the results.

Run it after `hotel-ml train` with `uv run python -m experiments.run_all`, or pass the names of the
experiments to run: `uv run python -m experiments.run_all calendar duplicates`.
"""

import argparse
import warnings
from collections.abc import Callable

import pandas as pd

from experiments import balancing, common, decisions, embeddings, encodings

EXPERIMENTS: dict[str, Callable[[pd.DataFrame], common.ExperimentResult]] = {
    "calendar": decisions.calendar,
    "duplicates": decisions.duplicates,
    "per_hotel": decisions.per_hotel,
    "tree_depth": decisions.tree_depth,
    "encodings": encodings.encodings,
    "balancing": balancing.balancing,
    "embeddings": embeddings.embeddings,
    "special_requests": lambda rows: decisions.special_requests(rows, tuned=common.champion_name()),
}


def main() -> None:
    """Run the requested experiments, one after the other."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", choices=[*EXPERIMENTS, []], default=[])
    arguments = parser.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

    development = common.load_development()
    for name in arguments.names or EXPERIMENTS:
        common.save(EXPERIMENTS[name](development))


if __name__ == "__main__":
    main()
