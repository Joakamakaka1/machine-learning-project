# Predicción de cancelaciones de reservas de hotel

**Autor:** Joaquín Castro Salas. **Repositorio:** <https://github.com/Joakamakaka1/machine-learning-project>

Sistema local que entrena, evalúa y compara modelos para predecir, **al confirmar una reserva**, si se va a cancelar, y sirve el mejor con una API y una interfaz. Es la práctica final del módulo de Machine Learning y Deep Learning del máster.

**Resultado:** un Random Forest calibrado obtiene en el test de 2017 un ROC-AUC de 0.8175, con un intervalo del 95 % de [0.8058, 0.8297]. Con filas únicas baja a 0.7608, y con `No Deposit`, a 0.7529. Un primer XGBoost con partición aleatoria daba 0.9524, y este proyecto explica por qué esa cifra no era creíble.

## Problema y datos

Una cancelación tardía deja una habitación vacía que ya no se puede vender. Se trata como clasificación binaria de `is_canceled`, en el único momento defendible con este archivo: al confirmar la reserva, porque el CSV guarda el valor final de cada reserva, no su historial.

Los datos son los del enunciado: 119 390 reservas de dos hoteles portugueses, con llegadas entre julio de 2015 y agosto de 2017.

> Antonio, N., de Almeida, A. y Nunes, L. (2019). *Hotel booking demand datasets*. Data in Brief, 22, 41-49. <https://doi.org/10.1016/j.dib.2018.11.126>. Licencia CC BY 4.0.

Quedan 118 563 filas tras limpiar, y entran 19 variables. La [data card](docs/data_card.md) justifica cada una.

## Resultados

Ejecución única sobre 40 509 reservas de 2017, sin reentrenar, con un [registro](outputs/final_evaluation/record.json) de fecha y commit. Se usa ROC-AUC como métrica principal porque no depende del umbral y no se deja engañar por la proporción de clases.

### Modelos

Con el umbral servido de 0.25, que sale de costes 1:3 supuestos, y con un umbral fijo de 0.5 para comparar con otros trabajos.

| Modelo | ROC-AUC | IC 95 % | PR-AUC | Accuracy, 0.25 | F1, 0.25 | Accuracy, 0.5 | F1, 0.5 |
|---|---|---|---|---|---|---|---|
| **Random Forest (campeón)** | 0.8175 | [0.8058, 0.8297] | 0.7608 | 0.5698 | 0.6294 | 0.7484 | 0.6695 |
| XGBoost | 0.8164 | [0.8046, 0.8284] | 0.7564 | 0.6062 | 0.6450 | 0.7455 | 0.6724 |
| MLP | 0.8039 | [0.7907, 0.8178] | 0.7654 | 0.6809 | 0.6696 | 0.7422 | 0.6406 |
| Árbol de decisión | 0.8040 | [0.7921, 0.8169] | 0.7147 | 0.6240 | 0.6510 | 0.7310 | 0.6861 |
| Regresión logística | 0.7956 | [0.7809, 0.8096] | 0.7353 | 0.6424 | 0.6517 | 0.7363 | 0.5882 |
| Dummy (baseline) | 0.5000 | [0.5000, 0.5000] | 0.3883 | 0.3883 | 0.5594 | 0.6117 | 0.0000 |

### Segmentos del campeón

**La cifra global nunca va sola.** `Non Refund` casi siempre se cancela, y las filas repetidas ayudan.

| Filas analizadas | Filas | Prevalencia | ROC-AUC | PR-AUC | Brier | Precisión | Recall | F1 | Coste 1:3 |
|---|---|---|---|---|---|---|---|---|---|
| Todas | 40 509 | 0.3883 | 0.8175 | 0.7608 | 0.1627 | 0.4729 | 0.9409 | 0.6294 | 0.4761 |
| Únicas | 29 667 | 0.3300 | 0.7608 | 0.5820 | 0.1807 | 0.4082 | 0.9253 | 0.5665 | 0.5166 |
| `No Deposit` | 36 356 | 0.3187 | 0.7529 | 0.5456 | 0.1810 | 0.3926 | 0.9199 | 0.5503 | 0.5301 |
| `Non Refund` | 4 126 | 0.9981 | 0.9998 | 1.0000 | 0.0010 | 0.9981 | 1.0000 | 0.9990 | 0.0019 |
| `City Hotel` | 27 396 | 0.4262 | 0.8201 | 0.7917 | 0.1644 | 0.4851 | 0.9670 | 0.6461 | 0.4797 |
| `Resort Hotel` | 13 113 | 0.3091 | 0.8020 | 0.6440 | 0.1592 | 0.4375 | 0.8658 | 0.5813 | 0.4685 |

### De 0.95 a 0.82

| Escalón | ROC-AUC |
|---|---|
| Partición aleatoria, variables de partida | 0.9524 |
| Partición temporal, variables de partida | 0.8683 |
| Partición temporal, sin `country` ni `booking_changes` | 0.8184 |
| Modelo oficial: 19 variables, sin calendario ni variables acumuladas | 0.8175 |

### Conclusiones

- **La cifra creíble es 0.82, no 0.95.** Coincide con la validación en 2016, de 0.8172 ± 0.0231, y con lo que el [protocolo](docs/protocolo.md) anticipaba antes de ejecutar el test.
- **2017 no era virgen:** una auditoría previa lo había visto, y de ella salen `ADR_UPPER_LIMIT`, `min_frequency=50` y las rejillas de búsqueda. Ninguna otra decisión sale de ahí.
- **Random Forest gana por la regla de 1 error estándar**, no por ser claramente mejor: empata con XGBoost, con una diferencia pareada de [-0.0033, +0.0008].
- **Lo que el test señala no se corrige, porque el modelo está congelado.** Con el umbral 0.25 se avisa del 77.3 % de las reservas; con costes 1:3 el campeón es el más caro de los cinco modelos, con 0.4761 por reserva; y la mejor PR-AUC es la de la MLP, con 0.7654.
- **Los costes son supuestos, no datos del hotel.** La interfaz permite probar otros.

Las tablas completas, las figuras y la reflexión crítica están en el [informe final](docs/informe_final.md), también en [PDF](docs/informe_final.pdf).

## Puesta en marcha

**Python 3.12** (`>=3.12,<3.13`; el modelo se empaquetó con 3.12.15). Docker Desktop solo hace falta para `poe up` y `poe report`.

### Con `uv` (recomendado)

Instala las versiones exactas de `uv.lock`, y Python 3.12 si no lo tienes.

```bash
uv sync --all-extras
```

### Con `python -m venv` y `pip`

Con un Python 3.12 ya instalado. `requirements.txt` fija todas las versiones e instala el proyecto.

```bash
python -m venv .venv
```

En Windows:

```bash
.venv\Scripts\activate
```

En Linux o macOS:

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

Con este entorno, los comandos son los mismos sin `uv run`.

### Comprobar que funciona

```bash
uv run poe check
```

Ejecuta ruff, mypy y todos los tests, con una cobertura mínima del 80 %. Tarda unos minutos.

## Cómo se ejecuta

### La solución entera, con Docker

```bash
uv run poe up
```

| Servicio | Dirección |
|---|---|
| Interfaz | <http://127.0.0.1:8501> |
| API, con su documentación | <http://127.0.0.1:8000/docs> |
| MLflow | <http://127.0.0.1:5000> |

Se detiene con `uv run poe down`.

`docker compose up` **sirve el campeón versionado en `models/champion/`**, sin entrenar nada, y comprueba su hash al arrancar. **El MLflow de Docker empieza vacío**, porque su volumen es nuevo: los runs reales están en `mlflow.db` y `mlartifacts/`, en local e ignorados por git. Para verlos:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

Y se abre <http://127.0.0.1:5001>. En un clon nuevo no hay `mlflow.db`: se crea al entrenar.

### Sin Docker

```bash
uv run poe api
```

```bash
uv run poe ui
```

### El pipeline, con `hotel-ml`

| Comando | Qué hace |
|---|---|
| `hotel-ml tune` | Busca hiperparámetros con las ventanas de 2016 |
| `hotel-ml train` | Selecciona el modelo, calibra, fija el umbral y empaqueta el campeón |
| `hotel-ml run-all` | `tune` y `train`, sin tocar el test |
| `hotel-ml train-scenario` | Entrena el escenario informativo |
| `hotel-ml rehearse` | Ensaya `evaluate` con 2016T4, en otra carpeta |
| `hotel-ml evaluate` | Ejecuta el test de 2017, una sola vez |
| `hotel-ml explain` | Importancia de las variables, tras el test |
| `hotel-ml audit` | Auditoría de la inflación, tras el test |
| `hotel-ml figures` | Figuras y valor de negocio, desde los resultados guardados |

Se ejecutan con `uv run hotel-ml <comando>`, y `--help` da sus opciones.

**El modelo está congelado.** `train`, `run-all` y `evaluate` se niegan a sustituirlo o a repetir el test sin `--force`. `tune` no tiene esa protección y reescribe `models/best_params.json` y `outputs/validation/`. Para probar el pipeline sin tocar nada, usa otras carpetas:

```powershell
$env:HOTEL_ML_MODELS_DIR = "$env:TEMP\hotel-demo\models"
$env:HOTEL_ML_OUTPUTS_DIR = "$env:TEMP\hotel-demo\outputs"
uv run hotel-ml run-all --profile fast --no-tracking
```

`--profile fast` usa versiones pequeñas de los modelos, sin MLP, y solo sirve para comprobar el flujo.

### Tareas de `poe`

| Tarea | Qué hace |
|---|---|
| `poe check` | ruff, formato, mypy y tests |
| `poe test-fast` | Los tests que no necesitan el extra de entrenamiento |
| `poe audit` | `pip-audit` sobre las dependencias |
| `poe up`, `poe down` | Levanta y detiene Docker |
| `poe api`, `poe ui` | API e interfaz en esta máquina |
| `poe notebooks` | Ejecuta los notebooks y guarda su salida |
| `poe report` | Genera `docs/informe_final.pdf`, con pandoc en un contenedor |
| `poe requirements` | Regenera `requirements.txt` desde `uv.lock` |

Los experimentos de diseño se ejecutan con `uv run python -m experiments.run_all`, y solo leen llegadas anteriores a 2017.

### Configuración

No hace falta ninguna, y no hay secretos. Si quieres cambiar algo, usa estas variables de entorno o un archivo `.env`: `HOTEL_ML_DATA_DIR`, `HOTEL_ML_MODELS_DIR`, `HOTEL_ML_OUTPUTS_DIR`, `HOTEL_ML_API_URL`, `HOTEL_ML_MLFLOW_TRACKING_URI` y `HOTEL_ML_MLFLOW_ARTIFACTS_DIR`.

## Arquitectura

```text
data/raw/*.csv.gz ─ hash y contrato ─ limpieza ─ partición temporal
                                                     │
        ┌─── desarrollo: llegadas anteriores a 2017 ─┴─ test: 2017 ───┐
        ▼                                                             │
experiments/ ─ decisiones de diseño                                   │
hotel-ml tune ─ RandomizedSearchCV con 4 ventanas ─ MLflow            │
hotel-ml train ─ regla 1-SE, calibración y umbral ─ models/champion   │
                                                     │                ▼
                                                     └─ hotel-ml evaluate, una vez
                                                                      │
models/champion + manifest ─ BookingPredictor ─ FastAPI ─ Streamlit   │
outputs/ ─ figuras, informe y notebooks ◄─────────────────────────────┘
```

Solo `evaluate` mide el test; `explain` y `audit` leen sus filas después, para interpretar. Los demás comandos cargan únicamente las filas de desarrollo. Las decisiones difíciles de revertir están en [`docs/adr/`](docs/adr/).

## Estructura

```text
src/hotel_cancellation/    Paquete instalable
  config.py  schema.py       Rutas y ventanas; fuente única de columnas
  data_loader.py             Hash, contrato, limpieza y particiones
  features.py  models.py     Preprocesado y catálogo de modelos
  model_trainer.py           Búsqueda, regla 1-SE y calibración
  evaluator.py               Métricas, costes, bootstrap y PSI
  pipeline.py  cli.py        Orquestación y comandos
  champion.py  tracking.py   Paquete del campeón y MLflow
  explain.py  audit.py  plots.py  Importancia, auditoría y figuras
  predictor.py  stored_evaluation.py  Inferencia y resultados guardados
  api/  ui/                  FastAPI y Streamlit
experiments/               Una decisión de diseño por experimento
scripts/                   Paridad entre la API y las predicciones guardadas
notebooks/                 EDA, auditoría de fugas y resultados, ejecutados
tests/                     Tests con cobertura mínima del 80 %
data/raw/                  Dataset comprimido, con hash y cita
models/champion/           Modelo y manifest, versionados
outputs/                   Métricas, figuras y resultados del test
docs/                      Informe, protocolo, cards, ADR, glosario y defensa
Dockerfile  compose.yaml   Imágenes de API e interfaz, y MLflow
pyproject.toml  uv.lock  requirements.txt  Dependencias
```

## Dónde se cumple cada requisito

| Requisito del enunciado | Dónde |
|---|---|
| Clasificación binaria con el dataset propuesto, justificada | [Informe](docs/informe_final.md), Justificación; [data card](docs/data_card.md) |
| 5 modelos: logística, árbol, Random Forest, boosting y red de Keras | [`models.py`](src/hotel_cancellation/models.py) |
| Métrica principal justificada | [Protocolo](docs/protocolo.md); [informe](docs/informe_final.md), Diseño |
| Matriz de confusión y curva ROC | [`matrices_confusion_modelos.png`](outputs/figures/matrices_confusion_modelos.png), [`roc.png`](outputs/figures/roc.png) |
| Pipeline: carga, preprocesado, entrenamiento, evaluación y selección | [`pipeline.py`](src/hotel_cancellation/pipeline.py), `hotel-ml run-all` y `evaluate` |
| Tabla comparativa de modelos | Este README; [informe](docs/informe_final.md), Resultados |
| Curvas comparativas e importancia de variables | [`outputs/figures/`](outputs/figures/) |
| EDA que motiva las decisiones | [`01_eda.ipynb`](notebooks/01_eda.ipynb); [informe](docs/informe_final.md), Análisis exploratorio |
| Módulos separados, funciones y clases reutilizables, Pipeline de scikit-learn | [`src/hotel_cancellation/`](src/hotel_cancellation/), [`features.py`](src/hotel_cancellation/features.py) |
| PEP 8, docstrings y organización | `uv run poe check` (ruff y mypy); Estructura, arriba |
| `.gitignore` y `requirements.txt` con versiones | [`.gitignore`](.gitignore), [`requirements.txt`](requirements.txt) |
| README: autores, problema, entorno y ejecución, resultados | Este README |
| Informe: roles, justificación, EDA, diseño, resultados y reflexión | [`informe_final.md`](docs/informe_final.md), [PDF](docs/informe_final.pdf) con la URL del repositorio |
| Defensa de 30 minutos ejecutando el código | [`defensa.md`](docs/defensa.md) |

| Bonus del enunciado | Dónde | Cómo se usa |
|---|---|---|
| API con FastAPI | [`api/`](src/hotel_cancellation/api/) | `/predict`, `/predict/batch` y `/evaluate`, sin `/train` ([ADR 0006](docs/adr/0006-sin-endpoint-train.md)) |
| Embeddings personalizados | [`experiments/embeddings.py`](experiments/embeddings.py) | MLP con embeddings categóricos: solo como experimento, porque aporta +0.0010 |
| Optimización de hiperparámetros | [`model_trainer.py`](src/hotel_cancellation/model_trainer.py) | `RandomizedSearchCV` con ventanas temporales; `hotel-ml tune` |
| Registro de experimentos | [`tracking.py`](src/hotel_cancellation/tracking.py) | MLflow con runs anidados, firma y alias `@champion` sobre el modelo evaluado |
| Balanceo de clases | [`experiments/balancing.py`](experiments/balancing.py) | SMOTE, undersampling y pesos de clase: medidos, y sin mejora; el campeón no los usa |
| Interpretabilidad | [`explain.py`](src/hotel_cancellation/explain.py) | `feature_importances_`, permutación y SHAP; `hotel-ml explain` |
| Interfaz visual | [`ui/`](src/hotel_cancellation/ui/) | Streamlit, que consume la API |

**Autoría:** el enunciado pide parejas. Este proyecto tiene un autor único, con todos los roles, como recoge el [informe](docs/informe_final.md).

## Documentación

- [Informe final](docs/informe_final.md) y su [PDF](docs/informe_final.pdf).
- [Protocolo de evaluación](docs/protocolo.md), fijado antes de comparar modelos.
- [Model card](docs/model_card.md) y [data card](docs/data_card.md).
- [ADR](docs/adr/): las decisiones difíciles de revertir.
- [Glosario](docs/glosario.md), pensado para aprender.
- [Guion de la defensa](docs/defensa.md).
- [Enunciado](docs/enunciado.md).
