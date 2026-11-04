# El paquete se organiza por etapas, con `models.py` fijo

`src/hotel_cancellation/` agrupa sus módulos en `data/`, `training/`, `evaluation/` y `serving/`, para que cada etapa del flujo tenga un sitio. En la raíz solo quedan `cli.py`, que las llama; `config.py` y `champion.py`, que usan todas; y `models.py`. Los `__init__.py` no importan nada, para que la imagen de la API siga sin TensorFlow ni MLflow.

**`models.py` no se mueve.** `models/candidates/mlp.joblib` guarda la ruta `hotel_cancellation.models.KerasMLPClassifier`, y su hash está registrado en `models/candidates.json`. Moverlo rompería `evaluate` y `rehearse`, porque [pickle necesita importar la clase](https://scikit-learn.org/stable/model_persistence.html), y regenerar el archivo cambiaría lo evaluado.

**Sin `configs/` en YAML.** Fechas, ventanas y rejillas son constantes de Python: el protocolo las fijó antes de comparar modelos, los tests las protegen y no hay configuraciones alternativas entre las que elegir. Lo que sí cambia de una máquina a otra, las rutas y las direcciones, se configura con variables `HOTEL_ML_*`, como recoge [`.env.example`](../../.env.example).

**Excepción a sacar los artefactos de git.** Se versionan el dataset (1.15 MB), el campeón (8.7 MB), `outputs/` (2.5 MB) y los notebooks con sus salidas (1.1 MB), para que un clon funcione solo y se pueda leer sin ejecutar nada. Dejaría de valer si un archivo llegara a decenas de MB o si los datos cambiaran con frecuencia. Entonces convendría DVC.

**Considerado:** seguir la plantilla de [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/opinions/) completa, que ella misma presenta como adaptable. Se descartan `data/interim`, `data/processed` y `data/external`, porque la limpieza ocurre en memoria; `reports/`, porque ya lo cubren `outputs/` y `docs/`; `utils/`, porque no tendría contenido propio; y el `Makefile`, porque `poe` ya hace ese papel y funciona igual en Windows y Linux.

**Coste aceptado:** `models.py` queda fuera de `training/`, donde encajaría por su contenido.
