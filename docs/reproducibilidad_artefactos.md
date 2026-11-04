# Reproducibilidad de los artefactos

Al reorganizar el repositorio, tres ejecuciones de `hotel-ml run-all --profile full --no-tracking` en otra carpeta dieron siempre el hash registrado del campeón (`41d3a285…`). En cambio, el escenario dio `09c5ffcc…` frente al `b198522e…` de `models/candidates.json`, y `mlp.joblib` dio un hash distinto cada vez. Este documento explica ambas diferencias.

| Pregunta | Respuesta |
| --- | --- |
| ¿Por qué `run-all` no da `b198522e…`? | Porque se registró con otra vía, `train-scenario`, que carga el campeón antes de entrenar. Esa vía sigue dando `b198522e…` con el código actual. |
| ¿Son distintos los dos modelos del escenario? | No. Su estado es idéntico, y solo cambia cómo pickle escribe unas cadenas. |
| ¿Cambia alguna cifra de `scenario_special_requests.csv`? | No. |
| ¿Por qué cambia el hash de `mlp.joblib`? | Por tres metadatos del archivo `.keras`. Los pesos son idénticos byte a byte. |
| ¿Cambia alguna cifra publicada del MLP? | No. |

Un SHA-256 de estos archivos identifica un fichero, no un modelo. Nada congelado cambió, y lo encontrado se declara sin corregirlo.

## El escenario

### Lo que no es la causa

- **El código.** Entre `185aa76` y `d293d5f`, siete commits tocan el entrenamiento y los datos (`94d2c01`, `7a4ca3b`, `4c27f73`, `e7a6c54`, `8ef6e29`, `359c57b`, `e998cb9`). Mueven módulos, cambian imports y añaden cargadores, constantes y funciones que este entrenamiento no usa. El cuerpo de `fit_special_requests_scenario` no cambió, y en `_fit_and_save_scenario` y `train_scenario` solo cambian dos referencias y un docstring.
- **Las dependencias.** `uv.lock` solo añade herramientas de desarrollo, como `pre-commit`, y sube la versión del propio paquete. scikit-learn, pandas, numpy y joblib siguen en las versiones del manifiesto.

### La causa

El escenario registrado no salió de `train` ni de `run-all`. El campeón y los candidatos se guardaron a las 22:40 del 2 de noviembre, con `d0b5caa`, cuyo `train` aún no entrenaba el escenario. El comando `train-scenario` llegó con `88df552`, y el escenario y el registro llevan la hora 23:18:06.

Las dos vías ajustan el mismo modelo, pero `train_scenario` llama antes a `champion.load_champion`, y `train` no carga ningún `joblib`.

| Ejecución | Código | Vía | Hash del escenario | Bytes |
| --- | --- | --- | --- | --- |
| Registrada | sin fijar el commit exacto | `train-scenario` | `b198522e…` | 10 255 584 |
| Repetición | `185aa76` | `train-scenario` | `b198522e…` | 10 255 584 |
| Repetición | `d293d5f` | `train-scenario` | `b198522e…` | 10 255 584 |
| Repetición, dos veces | `d293d5f` | `train` | `09c5ffcc…` | 10 251 154 |
| Durante la reorganización, tres veces | `a50656a`, `d293d5f` | `run-all` | `09c5ffcc…` | no consta |

Para aislar la causa, se ajustó el escenario en un proceso nuevo cambiando solo lo que ocurre antes:

| Antes de ajustar | Hash | Bytes |
| --- | --- | --- |
| Nada | `09c5ffcc…` | 10 251 154 |
| Importar Keras | `09c5ffcc…` | 10 251 154 |
| Ajustar el Random Forest servido | `09c5ffcc…` | 10 251 154 |
| Cargar el campeón | `b198522e…` | 10 255 584 |
| Cargar `mlp.joblib` | `570ed96e…` | 10 254 788 |

### El mecanismo

1. `BaseEstimator.__setstate__` hace `self.__dict__.update(state)`, así que las claves de un modelo cargado son cadenas nuevas, no las internadas de Python.
2. Después de esa carga, las instancias nuevas de esas clases usan claves distintas de las internadas. Se midió con las claves `criterion` y `max_features` de `DecisionTreeClassifier`, y `drop` y `dtype` de `OneHotEncoder`: son la cadena internada sin carga previa, y dejan de serlo tras cargar el campeón. Que lo transmita la tabla de claves compartidas de CPython es una inferencia, no una medición.
3. Pickle reconoce los objetos repetidos por identidad. Si `criterion` es un objeto como clave y otro en la tupla `estimator_params`, escribe la cadena dos veces en lugar de una referencia.
4. Esas cadenas desplazan el resto del flujo, joblib alinea cada array según su posición y la compresión da otro tamaño.

### Comparación de los dos ficheros

| Comparación entre `b198522e…` y `09c5ffcc…` | Resultado |
| --- | --- |
| Estado completo: preprocesado, 400 árboles y calibrador isotónico | 0 diferencias |
| Contenido de los 1626 arrays en el fichero, con los 400 de nodos | Idéntico |
| Instrucciones de pickle, resolviendo las referencias a cadenas | 59 624 en ambos, iguales |
| Cadenas escritas de forma literal | 198 frente a 178 |
| Bytes de alineación | 13 324 frente a 18 130 |
| Bytes sin comprimir | 30 540 603 frente a 30 540 379 |
| Predicciones en las 78 054 filas de desarrollo | Idénticas, diferencia máxima 0 |

Al comparar objetos ya cargados, el relleno de 7 bytes de cada nodo parece cambiar, incluso entre dos lecturas del mismo fichero. Es un artefacto de la carga: en ambos ficheros esos bytes son cero.

### Efecto en las cifras publicadas

Ninguno. `outputs/final_evaluation/record.json` registra `scenario_sha256` como `b198522e…`, y `models/scenario/special_requests.joblib` sigue teniendo ese hash. Como el estado de `09c5ffcc…` es idéntico, puntuaría lo mismo cualquier fila sin necesidad de leer las de 2017.

Estas cifras de `scenario_special_requests.csv` siguen siendo válidas. El escenario es informativo y usa una variable prohibida para el campeón.

| Subconjunto | Filas | ROC-AUC | PR-AUC | Brier |
| --- | --- | --- | --- | --- |
| Todas | 40 509 | 0.836947 | 0.78347 | 0.157724 |
| Filas únicas | 29 667 | 0.778156 | 0.61216 | 0.178484 |
| `No Deposit` | 36 356 | 0.779157 | 0.585625 | 0.175432 |
| `Non Refund` | 4 126 | 0.999681 | 0.999999 | 0.001059 |

## El MLP

Se compararon el fichero registrado y dos repeticiones con `train`:

| Fichero | Hash | Bytes | `date_saved` | `shared_object_id` | Fecha de `model.weights.h5` en el zip |
| --- | --- | --- | --- | --- | --- |
| Registrado | `bf1df560…` | 302 113 | 2026-11-02@22:40:55 | 2059102879232 | 2026-11-02 22:40:54 |
| Repetición 1 | `d05a0a9d…` | 302 109 | 2026-11-04@18:15:51 | 1755481666960 | 2026-11-04 18:15:50 |
| Repetición 2 | `b95f2df4…` | 302 107 | 2026-11-04@18:24:48 | 1673996206544 | 2026-11-04 18:24:48 |

`KerasMLPClassifier.__getstate__` guarda los bytes de un archivo `.keras`, que es un zip. Tres cosas de ese archivo cambian en cada guardado:

- **`metadata.json`:** `date_saved` es la hora del guardado.
- **`config.json`:** `shared_object_id` aparece en la política de tipos del modelo y de sus capas. Keras lo calcula con `id()`, así que es una dirección de memoria del proceso.
- **El zip:** `model.weights.h5` lleva la fecha de su fichero temporal.

Los pesos no cambian. `model.weights.h5` tiene el mismo SHA-256 en los tres (`8421b983…`, 338 672 bytes), también entrenado dos días después. El estado de la clase coincide, con `epochs_trained_` en 8, y las predicciones en las 78 054 filas de desarrollo son idénticas.

### Efecto en las cifras publicadas

Ninguno. Las medidas sobre 2016T4 de `outputs/validation/calibration.json` coinciden en todos sus decimales con las de las repeticiones, y el test se midió con esos mismos pesos. Estas son las cifras del MLP en `metrics.csv`, con el umbral de decisión, y en `calibration.json`:

| Subconjunto | ROC-AUC en 2017 | ROC-AUC en 2016T4 |
| --- | --- | --- |
| Todas | 0.803934 | 0.8124326316066044 |
| Filas únicas | 0.745187 | 0.7630345724666211 |
| `No Deposit` | 0.734938 | 0.7318204112446028 |
| `Non Refund` | 0.992533 | 0.989247311827957 |

## Qué implica

- **El registro sigue cumpliendo su función.** `models/candidates.json` demuestra que los ficheros no han cambiado desde que se registraron, y `evaluate` lo comprobó antes de leer el test.
- **El registro no sirve para comparar con un reentrenamiento en todos los casos.** Tras `run-all` o `train`, lo esperado es esto:

| Artefacto | Resultado esperado |
| --- | --- |
| Campeón y otros candidatos | Su hash registrado |
| Escenario | `09c5ffcc…`, con el mismo estado que `b198522e…` |
| `mlp.joblib` | Un hash nuevo cada vez, con `model.weights.h5` en `8421b983…` |

- **El hash del campeón también depende de la vía.** Ajustar el mismo Random Forest después de cargar el campeón dio `237b500f…` en lugar de `41d3a285…`, con el mismo contenido en el fichero. Es estable porque `train` y `run-all` no cargan ningún `joblib` antes de guardar.
- **`b198522e…` se puede reproducir.** Basta con `hotel-ml train-scenario` sobre una copia de `models/champion/` y `models/candidates.json`.

## Cómo se comprobó

```powershell
$env:HOTEL_ML_MODELS_DIR = "<otra carpeta>\models"
$env:HOTEL_ML_OUTPUTS_DIR = "<otra carpeta>\outputs"
uv run hotel-ml train-scenario                     # con la copia indicada arriba
uv run hotel-ml train --profile full --no-tracking # con una copia de best_params.json
```

Los límites de esta comprobación son estos:

- **Sin leer el test.** No se ejecutó `hotel-ml evaluate`, no se usó `--force` y no se leyeron filas de 2017 ni `predictions.csv.gz`. Las filas de 2017 solo pasan por la validación y limpieza de `load_development`, igual que en `run-all`, que las descarta antes de entrenar. Las predicciones se calcularon solo con filas de desarrollo.
- **`run-all` no se repitió aquí.** Sus hashes vienen de las ejecuciones hechas durante la reorganización, y `e4a9808`, con `scenario_frame` ya en `data_loader`, también da `09c5ffcc…`. `train` da el mismo `09c5ffcc…` usando la búsqueda guardada.
- **Una repetición se descartó.** `src/` se editó mientras se lanzaba una tercera ejecución de `train`. Dio los mismos hashes, salvo el de `mlp.joblib`, pero no se usa como prueba.
- **Las versiones y la máquina fueron las mismas.** No se comprobó si los hashes o los pesos se mantienen en otra máquina o con otras versiones.
- **Los scripts de comparación no están en el repositorio.** Se usaron como herramienta puntual.
