"""
Shim de compatibilidad para deserializar modelos entrenados con _TuningProgressPipeline.

Los modelos .pkl serializados con la antigua _TuningProgressPipeline necesitan
que la clase exista para que joblib.load pueda reconstruir el objeto.
Este archivo redirige la clase a sklearn.pipeline.Pipeline estándar.

Se puede eliminar una vez se reentrenen todos los modelos con el código refactorizado.
"""

from sklearn.pipeline import Pipeline as _TuningProgressPipeline  # noqa: F401
