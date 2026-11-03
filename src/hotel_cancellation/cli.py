"""Command line of the project: run `hotel-ml --help` to see every command.

Only `evaluate` measures the test period, after checking everything it needs, and it refuses to
repeat. `explain` and `audit` read its rows once it has run, only for interpreting. The others use
`load_development`, which drops its rows.
"""

import argparse
import logging
import tempfile
from collections.abc import Sequence
from pathlib import Path

from hotel_cancellation import champion, config
from hotel_cancellation.data import data_loader
from hotel_cancellation.evaluation import evaluator, final_evaluation
from hotel_cancellation.training import model_trainer, pipeline, tracking

logger = logging.getLogger(__name__)

_FORCE_HELP = "Cambia lo congelado aunque el test ya se haya ejecutado, y hay que declararlo."


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
        command.add_argument("--force", action="store_true", help=_FORCE_HELP)
        if name == "tune":
            command.add_argument(
                "--only", nargs="+", help="Busca solo estos modelos y conserva los demás."
            )

    description = "Entrena el escenario informativo del campeón y registra su hash."
    scenario = commands.add_parser("train-scenario", help=description, description=description)
    scenario.add_argument("--force", action="store_true", help=_FORCE_HELP)

    description = "Ensaya evaluate completo con una ventana de 2016, en un directorio temporal."
    rehearse = commands.add_parser("rehearse", help=description, description=description)
    rehearse.add_argument("--window", default="2016Q4", help="Ventana usada como pseudotest.")
    rehearse.add_argument("--output", type=Path, help="Directorio de salida.")
    rehearse.add_argument(
        "--replicates", type=int, default=evaluator.BOOTSTRAP_REPLICATES, help="Réplicas bootstrap."
    )

    evaluate = commands.add_parser(
        "evaluate",
        help="Ejecuta el test de 2017, una sola vez.",
        description="Ejecuta el test de 2017 con los modelos guardados, sin reentrenar.",
    )
    evaluate.add_argument("--force", action="store_true", help="Repite el test y lo declara.")
    evaluate.add_argument(
        "--replicates", type=int, default=evaluator.BOOTSTRAP_REPLICATES, help="Réplicas bootstrap."
    )

    description = "Calcula la importancia de las variables sobre 2017, tras el test."
    commands.add_parser("explain", help=description, description=description)
    description = "Reproduce la auditoría de la inflación en cuatro escalones, tras el test."
    commands.add_parser("audit", help=description, description=description)
    description = "Genera las figuras y el valor de negocio desde los resultados guardados."
    commands.add_parser("figures", help=description, description=description)
    return parser


def _explain(settings: config.Settings) -> None:
    # The check comes first, because the rows of the test must not be read if it fails.
    champion.assert_test_executed(settings)
    from hotel_cancellation.evaluation import explain

    explain.explain(data_loader.load_test(), settings=settings)
    logger.info("Importance in %s", settings.interpretation_dir)


def _audit(settings: config.Settings) -> None:
    champion.assert_test_executed(settings)
    from hotel_cancellation.evaluation import audit

    audit.audit_inflation(data_loader.load_cleaned(), settings=settings)
    logger.info("Audit in %s", settings.audit_dir)


def _figures(settings: config.Settings) -> None:
    champion.assert_test_executed(settings)
    from hotel_cancellation.evaluation import plots

    logger.info("Figures: %s", [path.name for path in plots.save_all(settings)])


def _evaluate(arguments: argparse.Namespace, settings: config.Settings) -> None:
    # The checks come first, because the rows of the test must not be read if they fail.
    final_evaluation.prepare_evaluation(settings, force=arguments.force)
    final_evaluation.evaluate(
        data_loader.load_test(),
        data_loader.load_development(),
        settings=settings,
        replicates=arguments.replicates,
        force=arguments.force,
    )
    logger.info("Results in %s", settings.final_evaluation_dir)


def _rehearse(arguments: argparse.Namespace, settings: config.Settings) -> None:
    output = arguments.output or Path(tempfile.mkdtemp(prefix="hotel-ml-rehearsal-"))
    folder = final_evaluation.rehearse(
        settings=settings,
        window=arguments.window,
        output_dir=output,
        replicates=arguments.replicates,
    )
    logger.info("Rehearsal in %s. Its figures do not count as an evaluation.", folder)


def _train(arguments: argparse.Namespace, settings: config.Settings) -> None:
    tracker: tracking.Tracker = (
        tracking.NullTracker()
        if arguments.no_tracking
        else tracking.MlflowTracker(settings.mlflow_tracking_uri, settings.mlflow_artifacts_dir)
    )
    development = data_loader.load_development()
    common = {
        "settings": settings,
        "profile": arguments.profile,
        "seed": arguments.seed,
        "tracker": tracker,
        "force": arguments.force,
    }
    if arguments.command == "tune":
        pipeline.tune(development, only=arguments.only, **common)
    elif arguments.command == "train":
        pipeline.train(development, **common)
    else:
        pipeline.run_all(development, **common)


def main(argv: Sequence[str] | None = None) -> int:
    """Run a command and return its exit code."""
    arguments = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    settings = config.get_settings()
    try:
        if arguments.command == "evaluate":
            _evaluate(arguments, settings)
        elif arguments.command == "rehearse":
            _rehearse(arguments, settings)
        elif arguments.command == "train-scenario":
            pipeline.train_scenario(
                data_loader.load_development(), settings=settings, force=arguments.force
            )
        elif arguments.command == "explain":
            _explain(settings)
        elif arguments.command == "audit":
            _audit(settings)
        elif arguments.command == "figures":
            _figures(settings)
        else:
            _train(arguments, settings)
    except (
        champion.TestAlreadyEvaluatedError,
        champion.EvaluationNotReadyError,
        champion.ChampionIntegrityError,
        champion.TestNotExecutedError,
    ) as error:
        logger.error("%s", error)
        return 1
    return 0
