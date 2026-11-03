# Guion de la defensa

Treinta minutos como máximo: se ejecuta el código en directo y se responde a preguntas sobre las decisiones. El guion deja unos 10 minutos para las preguntas.

**El mensaje:** el modelo no es lo más interesante. Lo es saber qué mide cada cifra: no 0.95, sino 0.82 medida una sola vez sobre 2017, y 0.79 con las reservas confirmadas tras el despliegue, con todo lo que señala declarado aunque deje mal al modelo elegido. Y queda un supuesto sin comprobar: que las variables sean las de ese momento.

Comandos probados en PowerShell, en Windows, con la duración que tuvieron allí.

## Antes de empezar

Con Docker Desktop encendido y unos minutos de margen. `poe up` tarda más si hay que reconstruir las imágenes.

```powershell
$env:PYTHONUTF8 = "1"
uv sync --all-extras
uv run poe up
```

En una segunda terminal, para los runs reales de MLflow:

```powershell
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

**Deja abiertos:** la interfaz en <http://127.0.0.1:8501>, la API en <http://127.0.0.1:8000/docs>, MLflow en <http://127.0.0.1:5001>, el [PDF del informe](informe_final.pdf) y el notebook [`03_resultados`](../notebooks/03_resultados.ipynb).

**No ejecutes** `hotel-ml evaluate`, que no se repite. Sin `--force`, `tune`, `train` y `run-all` se niegan a tocar lo congelado; para enseñarlos, usa las carpetas temporales del paso 2.

## En directo

### 1. El problema y el pipeline (3 min)

```powershell
uv run hotel-ml --help
```

Explica los comandos en orden: `tune`, `train`, `evaluate`, y que solo `evaluate` mide el test. Enseña [`schema.py`](../src/hotel_cancellation/schema.py), con las 19 variables y las 12 prohibidas.

### 2. El flujo completo, sin tocar nada (3 min)

Tarda unos 40 segundos. Escribe en una carpeta temporal, sin MLflow.

```powershell
$env:HOTEL_ML_MODELS_DIR = "$env:TEMP\hotel-demo\models"
$env:HOTEL_ML_OUTPUTS_DIR = "$env:TEMP\hotel-demo\outputs"
uv run hotel-ml run-all --profile fast --no-tracking
Get-Content "$env:TEMP\hotel-demo\outputs\validation\selection.csv"
Remove-Item Env:HOTEL_ML_MODELS_DIR, Env:HOTEL_ML_OUTPUTS_DIR
```

**Avisa antes:** `fast` usa modelos pequeños y sin MLP, solo para comprobar el flujo. Por eso puede ganar la regresión logística. Es la regla de 1 error estándar eligiendo al más simple, y demuestra que no está amañada a favor de Random Forest.

### 3. Los experimentos reales en MLflow (2 min)

En <http://127.0.0.1:5001>, abre el experimento `hotel-cancellation`: cada modelo con sus runs anidados. Luego, en *Models*, `hotel-cancellation` con el alias `@champion` y su `sha256`, el mismo del manifest.

Cuéntalo: el MLflow de Docker en el puerto 5000 está vacío, porque los runs reales están en `mlflow.db`, en local.

### 4. El test es único (2 min)

Tarda unos 30 segundos. Prueba sobre datos sintéticos que `evaluate` se niega a repetir y que el campeón queda protegido, sin tocar el test real.

```powershell
uv run pytest tests/test_final_evaluation.py -k "TestChampionIsFrozenAfterTheTest or refuses_to_repeat" --no-cov -q
Get-Content outputs\final_evaluation\record.json
```

Y la guarda de fugas, que tarda unos segundos:

```powershell
uv run pytest tests/test_features.py tests/test_champion.py -k forbidden --no-cov -q
```

### 5. Los resultados (3 min)

En el PDF o en el notebook `03_resultados`:

1. La auditoría de la inflación: 0.9524, 0.8683, 0.8184 y 0.8175.
2. La tabla de segmentos: 0.8175 con todas las filas, 0.7608 con únicas y 0.7529 con `No Deposit`.
3. La curva ROC y las matrices de confusión.
4. **Lo que el test señala y no se corrige.** Díselo tú antes de que te lo pregunten.
5. **La [segunda revisión](revision_datos_y_validacion.md).** 0.7917 con las reservas confirmadas desde 2017-01-01, 0.5541 con las cancelaciones tardías, y el supuesto de que las variables se conocen al confirmar.

### 6. La API (2 min)

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/v1/health
```

```powershell
$body = '{"hotel":"City Hotel","lead_time":45,"stays_in_weekend_nights":1,"stays_in_week_nights":2,"adults":2,"children":0,"babies":0,"meal":"BB","market_segment":"Online TA","distribution_channel":"TA/TO","is_repeated_guest":0,"previous_cancellations":0,"previous_bookings_not_canceled":0,"reserved_room_type":"A","deposit_type":"No Deposit","agent":"9","days_in_waiting_list":0,"customer_type":"Transient","adr":110.0}'
Invoke-RestMethod -Method Post -ContentType application/json -Body $body http://127.0.0.1:8000/api/v1/predict
```

Debe dar 0.4046 de probabilidad, `canceled`, y banda `medium`. Una petición con `country` da un error 422, porque esa variable no existe en el contrato.

Paridad entre lo guardado y lo servido, en unos 5 segundos:

```powershell
uv run python -m scripts.verify_served_predictions --rows 2000
```

Debe decir que coinciden las 2000 filas. Es un modelo creado en Windows, servido en Linux.

### 7. La interfaz (3 min)

En <http://127.0.0.1:8501>:

1. **Predicción:** una reserva normal, y luego con `deposit_type` en `Non Refund`, para ver el aviso.
2. **Umbral y coste:** mueve los costes, y enseña que el umbral 0.25 sale de un supuesto.
3. **Interpretabilidad:** `deposit_type` domina, y por qué no es una causa.
4. **Datos:** las variables excluidas y su motivo.

### Si algo falla

- **Docker no arranca:** `uv run poe api` y, en otra terminal, `uv run poe ui`.
- **Quieren ver todos los tests:** `uv run poe test-fast`, unos 2 minutos. `uv run poe check` tarda unos 8, así que llévalo ya ejecutado.
- **Quieren repetir la evaluación sin hacerlo de verdad:** `uv run hotel-ml rehearse --window 2016Q4 --replicates 50`, unos 25 segundos, sobre 2016T4 y en otra carpeta. Sus cifras no valen, porque el calibrador se ajustó en ese trimestre. Necesita los candidatos en `models/candidates/`, que están en tu máquina y no en un clon nuevo.

### Al terminar

```powershell
uv run poe down
```

## Preguntas probables

### Las incómodas

**¿Por qué 0.82 y no 0.95, como otros trabajos con estos datos?**
Porque 0.95 sale de evaluar mal. Con partición aleatoria, el 32.0 % del test ya está en train y se ve el futuro: da 0.9524. El mismo modelo con partición temporal baja a 0.8683, y sin `country` ni `booking_changes`, a 0.8184. La cifra coincide con la validación en 2016, de 0.8172 ± 0.0231.

**Entonces, ¿0.82 es lo que rinde al confirmar una reserva?**
No, y es lo que corrigió la segunda revisión. Corto por llegada, de modo que hay reservas confirmadas antes del corte. Con las confirmadas desde 2017-01-01 da 0.7917, y con `No Deposit`, 0.7540. Y hay un límite mayor: el artículo del dataset referencia los valores al día anterior a la llegada, así que no consta que sean los de ese momento. Con las variables definidas al crear o confirmar, en 2016 da 0.6866. El rendimiento real está en ese rango, y este archivo no permite saber dónde.

**Justificas el proyecto por la cancelación tardía. ¿La detecta?**
Casi no. Frente a las estancias, da 0.5541 con las canceladas hasta 7 días antes de llegar y 0.5170 con los `No-Show`, frente a 0.8739 con las de 31 días o más. Parte de esa diferencia es estructural, porque con poca antelación no se puede cancelar con un mes de margen. Para detectarla haría falta predecir cerca de la llegada, con el historial de la reserva.

**¿Has probado a mejorarlo?**
Sí. En la segunda revisión probé 50 cambios en datos, variables y modelos, de uno en uno, con una regla fijada antes de medir, y ninguno la cumple. Dando al modelo filas del mismo trimestre, algo imposible en producción, llega a 0.8325. Lo que más valor tendría es el registro de cambios de la reserva, y reentrenar con frecuencia.

**¿Por qué Random Forest, si empata con XGBoost?**
Porque empata. La regla de 1 error estándar los deja dentro a los dos, con 0.8172 y 0.8162, y a igualdad gana el más simple, en un orden fijado antes de comparar. En el test la diferencia va de [-0.0033, +0.0008]. No digo que sea mejor, digo que no hay razón para el más complejo.

**¿Por qué se mantiene un umbral malo, que avisa del 77.3 % de las reservas?**
Porque cambiarlo tras ver el test invalidaría la cifra. El umbral sale de una fórmula y de unos costes 1:3 supuestos, fijados antes, y da 16 498 falsas alarmas. Con umbrales entre 0.27 y 0.30, el coste bajaría a unos 0.43, y lo publico. Además, no lo vi venir: con esos mismos costes, en 2016 avisaba del 46 % al 63 %. En una versión real, lo decidiría con los costes del hotel y con datos nuevos.

**¿Qué pasa con `total_of_special_requests`? Con ella darías 0.8369.**
Queda fuera por cautela. El diagnóstico indica que crece con la vida de la reserva: en Online TA va del 24.0 % al 58.5 %, más que el control. Supera al control en seis de ocho cohortes, no en todas, y es observacional, así que no está demostrado. Por eso se fijó antes del test un escenario informativo, y publico los dos. Si no fuera fuga, estaría renunciando a esa mejora.

**Con costes 1:3 tu campeón es el más caro. ¿No has elegido mal?**
Por coste, sí: 0.4761, frente a 0.4489 de la MLP. La selección se hizo por ROC-AUC, como mandaba el protocolo, y el ROC-AUC ordena, pero no mide lo que cuesta decidir. Es la primera mejora que propongo. Cambiarlo ahora sería elegir mirando el test.

**La MLP tiene mejor PR-AUC. ¿Por qué no la eliges?**
La tiene: 0.7654 frente a 0.7608. La métrica principal se fijó antes, y en ROC-AUC queda en 0.8039. Con otra métrica principal, el orden habría cambiado, y por eso lo declaro.

**Tu accuracy con el umbral servido es 0.5698, menos que no avisar nunca (0.6117). ¿Cómo lo explicas?**
Porque ese umbral no busca acertar más, sino perder menos cancelaciones, bajo el supuesto de que cuestan tres veces más. Con 0.5 la accuracy sube a 0.7484. Por eso la accuracy no es la métrica principal.

**¿El test es de verdad ciego?**
No del todo, y lo declaro. Una auditoría previa vio 2017, y de ella salen `ADR_UPPER_LIMIT`, `min_frequency=50` y las rejillas de búsqueda. Ninguna otra decisión sale de ahí, y el modelo no cambió tras la ejecución. Después volví a leer sus predicciones guardadas en un análisis posterior, fijado de antemano, que no puntúa ninguna fila ni cambia nada. Un test estricto necesitaría datos nuevos.

**El rendimiento cae en verano. ¿Sirve el modelo?**
Baja de 0.8742 en enero a 0.7568 en julio, y `agent` tiene un PSI de 0.7591. El intervalo del bootstrap no lo cubre, y por eso publico el mes a mes. No se reentrena durante 2017 para que lo evaluado sea lo servido. En producción haría falta vigilarlo y reentrenar.

**El enunciado pide parejas y `/train`. ¿Por qué no los hay?**
Autor único, con todos los roles, como recoge el informe. **Confírmalo con el profesor antes de la defensa**: el enunciado pide comunicarlo si no hay pareja. `/train` lo cita como ejemplo: entrenar es pesado e inseguro en un servicio, y permitiría sustituir un modelo congelado ([ADR 0006](adr/0006-sin-endpoint-train.md)).

### Sobre datos y variables

**¿Por qué se predice al confirmar?**
Porque es cuando el hotel puede actuar, pero es una intención. Primero lo presenté como el único momento defendible, y era un error: el artículo del dataset referencia los valores al día anterior a la llegada, y no hay historial para reconstruir el valor al confirmar.

**¿Por qué fuera `country`, si predice tanto?**
Justo por eso. Antonio, de Almeida y Nunes (2019) documentan que el sistema rellena Portugal por defecto y el país real se corrige en el check-in, de modo que las canceladas conservan el valor por defecto.

**¿Por qué conservas `deposit_type`, si `Non Refund` casi siempre se cancela?**
Es discutible. El artículo la calcula con los pagos anteriores a la llegada o a la cancelación, así que no consta que se conozca al reservar, y no puedo descartar que sea en parte una fuga. Quitarla no lo resuelve: el ROC-AUC en 2016 solo baja de 0.8207 a 0.8072, porque otras variables recogen parte de su señal. Por eso no la leo como causa, y publico el rendimiento sin esas filas: `No Deposit` da 0.7529. El 0.9998 de `Non Refund` no dice nada útil, porque el 99.8 % se cancela.

**¿Por qué cortas por fecha de llegada?**
Porque el resultado de una reserva solo se conoce del todo al llegar. Con un corte por llegada, todo lo que hay en train ya está resuelto. Tiene un coste: en cada ventana hay reservas confirmadas antes del corte, y por eso publico aparte las confirmadas después.

**¿`adr` es seguro?**
No se puede demostrar. El diagnóstico detecta acumulación, no sobrescritura, y `adr` no crece con la vida de la reserva, pero podría reemplazarse: el artículo la calcula con las transacciones de alojamiento. Sus decimales no delatan el resultado, y quitarla apenas cambia nada (-0.0016 en 2016). Es una limitación declarada.

**¿Por qué no deduplicas?**
Lo medí: entrenar con filas únicas da 0.8116, frente a 0.8141. Pero las métricas se publican también con filas únicas.

### Sobre modelos y métricas

**¿Por qué ROC-AUC como principal?**
Está en la lista del enunciado, no depende del umbral, y no se deja engañar por la proporción de clases. Sus límites: no mide el coste ni si la probabilidad es correcta. Por eso van también PR-AUC, Brier, pendiente y coste.

**¿Por qué calibras, y por qué con isotónica?**
Porque el umbral se aplica a una probabilidad, y debe ser creíble. En 2016T4, el Brier baja de 0.1622 a 0.1592 con isotónica, y supera la mejora mínima de 0.001. Su pega son los escalones: el 10.05 % de las filas tiene probabilidad exactamente 1.

**¿Por qué la regla de 1 error estándar con solo cuatro ventanas?**
Porque con cuatro ventanas las diferencias pequeñas son ruido, y elegir la media más alta sobreajustaría la selección. Sé que cuatro ventanas estiman mal el error estándar, y por eso el empate se lee como que no hay evidencia para distinguirlos, no como que sean iguales.

**¿Y el balanceo?**
Medido, en [`balancing.py`](../experiments/balancing.py): SMOTE, undersampling y pesos no mejoran, y distorsionan la probabilidad. El campeón no los usa.

**¿Y los embeddings?**
Una MLP con embeddings categóricos da 0.8063, frente a 0.8053 con one-hot. Tan poco que no justifica el cambio: queda como experimento.

**¿Por qué no explicas cada predicción?**
Con Random Forest no hay una contribución barata por reserva en la API. La interpretación es global: `feature_importances_`, permutación y SHAP, y no indica causas.

**¿Por qué el modelo está en git?**
Para que lo evaluado sea lo servido y funcione sin entrenar. El manifest guarda hash y versiones, y la API los comprueba al arrancar ([ADR 0005](adr/0005-modelo-versionado-con-manifest.md)).
