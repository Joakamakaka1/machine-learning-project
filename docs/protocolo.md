# Protocolo de evaluación

Este protocolo se fija **antes de comparar ningún modelo**. Si hubiera que cambiarlo, se hará en un commit propio que explique el motivo, y el informe final lo declarará.

## Qué se evalúa

Clasificación binaria de `is_canceled` **al confirmar la reserva**, con las 19 variables de entrada de [`schema.py`](../src/hotel_cancellation/schema.py). La disponibilidad de cada variable se justifica en la [data card](data_card.md).

## Particiones

La evaluación oficial es temporal y corta por fecha de llegada. Nunca se usa una partición aleatoria: una parte importante del test aleatorio ya estaría en train por los duplicados exactos.

| Parte | Llegadas | Uso |
|---|---|---|
| Desarrollo | Anteriores a 2017-01-01 | Toda decisión: EDA, variables, hiperparámetros, modelo, calibración y umbral |
| Test | 2017-01-01 a 2017-08-31 | Una sola ejecución, al final |

## 2017 no es virgen

Se declara expresamente: **los datos de 2017 ya se han visto**.

- En la auditoría previa medí con ellos la inflación de las métricas: un XGBoost con las variables de partida, entrenado hasta 2016 y evaluado en 2017, dio un ROC-AUC de 0.868.
- También aparecen en estadísticas globales de esa auditoría.

Ninguna decisión sale de ahí. El EDA del proyecto se limita a llegadas anteriores a 2017. En este repositorio, las filas de 2017 solo se ven en la validación del contrato y en los recuentos de limpieza. Los valores permitidos se obtuvieron del CSV completo, y un test comprueba que todos aparecen ya antes de 2017.

El test, por tanto, no es estrictamente ciego, y el informe final lo dirá junto a la cifra.

## Ventanas

| Finalidad | Entrenamiento | Ajuste | Medida |
|---|---|---|---|
| Validación expansiva, 2016T1 | Anterior a 2016-01-01 | | 2016T1 |
| Validación expansiva, 2016T2 | Anterior a 2016-04-01 | | 2016T2 |
| Validación expansiva, 2016T3 | Anterior a 2016-07-01 | | 2016T3 |
| Validación expansiva, 2016T4 | Anterior a 2016-10-01 | | 2016T4 |
| Elegir la calibración | Anterior a 2016-07-01 | 2016T3 | 2016T4 |
| Campeón servido | Anterior a 2016-10-01 | 2016T4 | Test, 2017 |

Las filas se ordenan por llegada antes de entrenar, de modo que el `validation_split` de Keras valide con lo más reciente.

## Búsqueda de hiperparámetros

- `RandomizedSearchCV` con un máximo de 20 iteraciones por modelo. La MLP prueba 6 configuraciones.
- Las 4 ventanas expansivas se pasan como lista de índices por fecha.
- La búsqueda es multimétrica con `refit=False`, y las puntuaciones por ventana salen de `cv_results_`.

## Métricas

- **Principal:** ROC-AUC, que está en la lista del enunciado y no depende del umbral.
- **Secundarias:** PR-AUC, Brier, pendiente de calibración, coste, y accuracy, precisión, recall y F1. El F1 es siempre de la clase positiva.
- **Baseline:** `DummyClassifier`, que el campeón debe superar.
- El ROC-AUC global nunca va solo: se reporta con todas las filas, con filas únicas, con `No Deposit` y con `Non Refund`.

## Selección del modelo

Se aplica la regla de 1 error estándar sobre las diferencias pareadas por ventana. Entre los modelos que queden dentro, gana el más simple, en este orden: regresión logística, árbol, Random Forest, XGBoost y MLP. Si gana la regresión logística, se acepta.

Las cifras de la auditoría que motivaron cada decisión se repiten con el conjunto final de variables.

## Calibración

Se comparan sigmoide, isotónica y ninguna, ajustando en 2016T3 y midiendo en 2016T4. Si el Brier mejora menos de 0.001, no se calibra.

## Campeón servido

El modelo base se entrena con llegadas anteriores a 2016-10-01 y el calibrador se ajusta en 2016T4. Se evalúa en 2017 **sin reentrenar**, para que lo evaluado sea exactamente lo que se sirve.

## Umbral y costes

El umbral es `t* = C_FP / (C_FP + C_FN)` sobre la probabilidad calibrada. Los costes son **supuestos, y se declaran como tales**: tres escenarios `C_FP:C_FN` de 1:2, 1:3 y 1:5, con un contraste frente al coste empírico de 2016T4 y controles en la interfaz para probar otros.

## Incertidumbre

- Bootstrap por bloques `hotel × fecha de llegada`, con 2000 réplicas y pareado entre modelos, y sensibilidad por semana.
- El intervalo no cubre la deriva temporal, así que se publica también la dispersión entre ventanas, y el ROC-AUC y la prevalencia mes a mes.

## Informe

- Tabla comparativa con el baseline.
- Curvas ROC y PR comparativas, matrices de confusión legibles y calibración.
- Segmentos y deriva (PSI).
- Importancia de variables: `feature_importances_` de Random Forest, permutación y SHAP.
- Dos escenarios, con y sin `total_of_special_requests`, porque el diagnóstico de acumulación es observacional.

## Ejecución única del test

- `evaluate` deja constancia en `outputs/final_evaluation/`, con fecha y commit, y se niega a repetir sin `--force`.
- Después no se toca el modelo. Si algo obligara a hacerlo, se declara en el informe.
- La cifra se publica aunque decepcione. **Antes de ejecutarlo se espera un ROC-AUC en torno a 0.81**, el valor de la validación temporal en 2016, y puede salir menor por la deriva. Si el campeón no supera al baseline, se dirá.

## Enmienda 1 (2026-11-02), antes de ejecutar el test

**Motivo:** una revisión independiente del código encontró fallos en `evaluate` antes de que se ejecutara el test. Nadie ha visto todavía resultados de estos modelos en 2017. La enmienda no cambia el campeón, ni sus ventanas, ni su umbral.

### Escenario con `total_of_special_requests`

Random Forest con los hiperparámetros del campeón, las mismas ventanas y calibración isotónica, medido en 2017 dentro de la misma ejecución.

- Es **solo informativo** y no puede cambiar el campeón.
- Su modelo se entrena y se registra con hash antes del test.

### Interpretabilidad

La permutación y SHAP se calculan sobre 2017 una vez congelado el modelo, solo para interpretar. Ninguna decisión sale de ahí.

### Decisiones que salen de la auditoría previa

La auditoría había visto 2017, y de ella salen tres elecciones que afectan al modelo:

- `ADR_UPPER_LIMIT`, el corte de limpieza de `adr`.
- `min_frequency=50`, en la codificación de categorías.
- Las rejillas de búsqueda de hiperparámetros.

Se conservan, y el informe final lo declara junto a la cifra del test.

### Definiciones de las métricas

- **Pendiente de calibración:** se recortan las probabilidades a `[1e-3, 1 - 1e-3]` antes de calcular el logit. Con isotónica, la pendiente depende mucho de ese recorte, así que se informa también de la sensibilidad con `1e-6` y `1e-2`, y del porcentaje de filas con probabilidad exactamente 0 o 1.
- **Tablas ya generadas:** las de validación y experimentos anteriores a esta enmienda usaron un recorte de `1e-6`, y no se regeneran para no reentrenar. `outputs/experiments/calibration_sensitivity.csv` da la pendiente de cada modelo con los tres recortes.
- **Coste:** cada escenario se mide en su propio `t*`, igual que en validación. Además, se da el coste de los tres escenarios al umbral servido.
- **Métricas no definidas:** en un subconjunto con una sola clase, o sin predicciones positivas, valen `NaN`, y siempre van con el recuento de filas.
- **PSI:** las variables discretas o con empates masivos usan un bin por valor; las demás, cuantiles.

### Salvaguardas de la ejecución única

- **Hash de los modelos:** los SHA-256 de todos los candidatos y del escenario están en `models/candidates.json`, versionado. Antes de leer ninguna fila de 2017, `evaluate` falla si falta alguno, si un hash no coincide o si el del candidato ganador no es el del campeón.
- **Condiciones de la ejecución real:** exige el perfil `full`, el árbol de git limpio y los directorios por defecto.
- **Registro:** todo se calcula en memoria y el registro aparece a la vez que los resultados, de forma atómica. Un fallo no deja cifras visibles ni cuenta como ejecución.
- **Campeón protegido:** si existe el registro, `train` y `run-all` se niegan a sustituir al campeón sin `--force`.

### Ensayo general

`hotel-ml rehearse` ejecuta `evaluate` completo con 2016T4 como pseudotest, en un directorio temporal. Solo comprueba que el código funciona. Sus cifras no valen como evaluación, porque el calibrador del campeón se ajustó en ese mismo trimestre.
