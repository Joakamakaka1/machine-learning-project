# El modelo se versiona en git con un manifest

`models/champion/` guarda el `model.joblib` y un `manifest.json` con su hash, las versiones de las librerías, el umbral, las ventanas y las métricas de validación. Lo evaluado debe ser exactamente lo servido, y cualquiera que clone el repositorio tiene que poder usarlo sin entrenar.

La API comprueba el hash y las versiones al arrancar, y el artefacto solo usa clases de scikit-learn. Los demás candidatos no se versionan: `models/candidates.json` guarda sus hashes, que `evaluate` comprueba antes de leer ninguna fila de 2017.

**Considerado:** un registro de modelos como única fuente. Exigiría un servidor, y el proyecto es local. MLflow registra los experimentos, con el alias `@champion` apuntando a este mismo artefacto, pero no es de donde se sirve.

**Coste aceptado:** un binario en git, de unos 8.7 MB. Es asumible porque el modelo está congelado y no cambiará.

**Límite aceptado:** el hash vive junto al modelo, así que detecta un fichero corrupto o sustituido, pero no a quien cambie a la vez el modelo y el manifest. Cargar un `joblib` ejecuta código, de modo que solo debe cargarse un modelo de este repositorio. Para un proyecto local es suficiente; publicarlo exigiría firmar el artefacto o usar un formato sin ejecución de código.
