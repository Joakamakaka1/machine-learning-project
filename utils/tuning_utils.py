"""
Utilidades para la búsqueda de hiperparámetros.

Contiene _TuningProgressPipeline, una subclase de Pipeline que registra el
progreso fit a fit durante RandomizedSearchCV, mostrando % completado y ETA.
"""

import logging
import time

from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class _TuningProgressPipeline(Pipeline):
    """Subclase de Pipeline que loguea el progreso de cada fit durante la búsqueda.

    Usa estado de clase para que todas las instancias clonadas por sklearn
    compartan el mismo contador. Requiere n_jobs=1 en RandomizedSearchCV para
    que los logs lleguen al proceso principal (sin procesos hijo).
    """

    _count: int = 0
    _total: int = 0
    _start_time: float = 0.0

    @classmethod
    def reset(cls, total: int) -> None:
        """Reinicia el contador antes de iniciar una nueva búsqueda."""
        cls._count = 0
        cls._total = total
        cls._start_time = time.time()

    def fit(self, X, y=None, **fit_params):
        _TuningProgressPipeline._count += 1
        count = _TuningProgressPipeline._count
        total = _TuningProgressPipeline._total
        elapsed = time.time() - _TuningProgressPipeline._start_time
        pct = count / total * 100
        if count > 1:
            eta_sec = elapsed / count * (total - count)
            eta_str = f"{eta_sec / 60:.1f} min"
        else:
            eta_str = "calculando..."
        logger.info(
            "[%d/%d] %.0f%% completado — %.0fs transcurridos — ETA: %s",
            count,
            total,
            pct,
            elapsed,
            eta_str,
        )
        return super().fit(X, y, **fit_params)
