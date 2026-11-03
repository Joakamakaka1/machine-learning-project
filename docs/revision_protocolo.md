# Protocolo de la segunda revisión

Se fija **antes de ejecutar ningún experimento de esta revisión**. Complementa al [protocolo original](protocolo.md), que no cambia. Lo que salga de aquí no toca `models/champion/` ni `outputs/final_evaluation/`.

## Pregunta

¿Es sólido el trabajo con los datos y la validación? ¿Hay fugas o sesgos sin detectar? ¿Hay mejoras reales, o el dataset impone un techo?

## El test de 2017 ya está gastado

Se ejecutó una vez el 2026-11-02, con su cifra publicada y leída durante esta revisión. No existen más datos, de modo que **ningún resultado futuro sobre 2017 sería una prueba ciega**.

- Hasta el paso final, ningún código de esta revisión puntúa filas de 2017 ni lee sus etiquetas. La única excepción es recalcular la línea base desde las predicciones ya guardadas, que no genera información nueva sobre ningún modelo.
- Si aparece un candidato que justifique evaluarlo en 2017, el autor decide si se hace. Antes se fijarían por escrito el candidato, las métricas y la comparación, y su resultado se etiquetaría como evaluación posterior, no ciega.

## Datos

- Solo llegadas anteriores a 2017: 78 054 filas limpias.
- La limpieza de `clean` se conserva durante los experimentos, y cualquier cambio se prueba como un experimento más.
- Variables derivadas solo para analizar, que nunca entran en un modelo: `booking_date = arrival_date - lead_time` y `status_date`.
- `reservation_status` y `reservation_status_date` solo sirven para auditar, porque dependen del resultado. Las claves que se usan para purgar filas no pueden usarlas.

## Esquemas de validación

Todos usan origen móvil y expansivo por fecha de llegada: se entrena con lo anterior a cada origen.

| Esquema | Ventanas | Uso |
|---|---|---|
| `Q4` | Los 4 trimestres de 2016, los del protocolo original | Comparar con lo publicado |
| `M12` | Los 12 meses de 2016 | **Decidir**, por tener más parejas de diferencias y un error estándar mejor estimado |

Las ventanas de `M12` se solapan en sus datos de entrenamiento, así que no son independientes y su error estándar puede quedarse corto. Por eso una mejora tiene que verse también en `Q4`.

Además, y solo para diagnosticar el esquema:

- **Purga por tandas:** quitar del entrenamiento las filas de una tanda que también está en validación. Una tanda es la misma combinación de hotel, fecha de reserva, agente, segmento, depósito, tipo de cliente y habitación, todo conocido al reservar.
- **Reservado después del corte:** medir aparte las filas confirmadas tras el origen, que son las que un modelo desplegado en esa fecha vería al confirmarlas.
- **Horizonte:** medir cómo cae el rendimiento al alejar la validación del final del entrenamiento, porque el campeón se aplica hasta 11 meses después de su último dato de entrenamiento.
- **Contraste aleatorio:** `StratifiedKFold` y `GroupKFold` dentro de desarrollo, solo para medir cuánto infla la partición aleatoria.

## Métricas

- **Principal:** ROC-AUC, como en el protocolo original.
- **Acompañantes, en todas las tablas:** ROC-AUC con `No Deposit`, ROC-AUC de filas distintas, PR-AUC, log-loss y Brier.
- **Definición nueva de filas distintas:** se deduplica por las 19 variables de entrada, sin usar el objetivo, y con el objetivo medio de cada grupo como peso. La definición original incluye el objetivo en la clave y por eso depende del resultado. Se calculan las dos.

## Regla para aceptar un cambio

Cada experimento cambia **una sola cosa** frente a una referencia fija. Un cambio se acepta solo si cumple todo esto:

1. La diferencia media pareada en `M12`, en ROC-AUC total o con `No Deposit`, es de al menos **+0.002**.
2. Su intervalo t del 95 % excluye el cero, con corrección de Holm dentro de cada familia de comparaciones.
3. No empeora de forma significativa la otra métrica de esas dos, ni el log-loss.
4. Mantiene el signo en `Q4`.
5. Mantiene el signo con 5 semillas, si el modelo tiene azar.

El umbral de +0.002 se fija porque está en el orden de la semianchura del intervalo de la diferencia pareada del test ([-0.0033, +0.0008]), y por debajo de eso no se distingue del ruido.

Si nada lo cumple, la conclusión es que no hay mejora demostrable, y así se dirá.

## Orden

1. Auditoría del dataset y de sus etiquetas.
2. Fugas y problemas estructurales.
3. Diagnóstico del esquema de validación.
4. Escalera de modelos, desde los baselines triviales.
5. Tratamientos de datos, de uno en uno.
6. Variables: ablación por grupos de riesgo y variables nuevas calculadas solo con el pasado.
7. Búsqueda de hiperparámetros anidada, para medir el optimismo de la selección.
8. Calibración y umbral, en varias parejas de origen.
9. Análisis de errores, robustez y curva de aprendizaje.
10. Decisión sobre 2017.

## Reproducibilidad

Semilla 42, salvo en las pruebas de semillas, con las versiones de `uv.lock`. Cada paso se ejecuta con `uv run python -m experiments.review.<módulo>` y escribe en `outputs/review/`.
