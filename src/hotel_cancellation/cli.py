"""Command line: `hotel-ml tune|train|run-all`.

No command here reads the test period: `load_development` drops its rows.
"""

import argparse
import logging
from collections.abc import Sequence

from hotel_cancellation import config, model_trainer, pipeline, tracking


def build_parser() -> argparse.ArgumentParser:
    """Declare the commands."""
    parser = argparse.ArgumentParser(prog="hotel-ml", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, description in (
        ("tune", "Busca hiperparámetros con las ventanas expansivas de 2016."),
        ("train", "Selecciona el modelo, entrena los candidatos y empaqueta el campeón."),
        ("run-all", "Ejecuta tune y train, sin tocar el test de 2017."),
    ):
        command = commands.add_parser(name, help=description, description=description)
        command.add_argument("--profile", choices=("full", "fast"), default="full")
        command.add_argument("--seed", type=int, default=model_trainer.DEFAULT_SEED)
        command.add_argument(
            "--no-tracking", action="store_true", help="No registra nada en MLflow."
        )
        if name == "tune":
            command.add_argument(
                "--only", nargs="+", help="Busca solo estos modelos y conserva los demás."
            )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a command and return its exit code."""
    arguments = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    settings = config.get_settings()
    tracker: tracking.Tracker = (
        tracking.NullTracker()
        if arguments.no_tracking
        else tracking.MlflowTracker(settings.mlflow_tracking_uri, settings.mlflow_artifacts_dir)
    )
    development = pipeline.load_development()
    common = {
        "settings": settings,
        "profile": arguments.profile,
        "seed": arguments.seed,
        "tracker": tracker,
    }
    if arguments.command == "tune":
        pipeline.tune(development, only=arguments.only, **common)
    elif arguments.command == "train":
        pipeline.train(development, **common)
    else:
        pipeline.run_all(development, **common)
    return 0
