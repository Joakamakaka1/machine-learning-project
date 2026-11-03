# Convenciones del repositorio

Predicción de cancelaciones de reservas al confirmarlas. Lee el `README.md` para la estructura y los comandos, y `docs/protocolo.md` antes de tocar datos, modelos o métricas.

## Idiomas

- Código, identificadores, docstrings y comentarios: inglés.
- Documentación, notebooks, interfaz, mensajes al usuario y commits: español.

## Modelo congelado

- `models/champion/` y `outputs/final_evaluation/` no cambian. El SHA-256 de `model.joblib` es `41d3a28592ae93367b8a88ca80ed60bb1e5ac2356ec2344d4984c0770f13d7f1`.
- El test de 2017 ya se ejecutó, una sola vez: `hotel-ml evaluate` nunca se repite. Lo que señala se declara, no se corrige.
- Comprueba el flujo de entrenamiento en otra carpeta, con `HOTEL_ML_MODELS_DIR` y `HOTEL_ML_OUTPUTS_DIR`. `--force` cambia lo congelado: úsalo solo si el autor lo pide, y decláralo.
- Toda decisión nueva sale de las llegadas anteriores a 2017. Los notebooks y `outputs/eda/` también.

## Momento de predicción

Predecir al confirmar es una intención, no un hecho: el artículo del dataset referencia los valores al día anterior a la llegada. No escribas que una variable se conoce al confirmar sin respaldarlo, y lee `docs/revision_datos_y_validacion.md` antes de afirmar qué mide una cifra.

## Cifras

Copia cada cifra de `outputs/` o de `manifest.json`. `tests/test_documentation.py` compara las principales con el README y el informe. El ROC-AUC global va siempre con filas únicas, `No Deposit` y `Non Refund`. Los costes se nombran como supuestos.

## Comandos

- `uv run poe check` en verde antes de cada commit. Tarda unos 8 minutos.
- `uv run poe notebooks` tras cambiar los notebooks o lo que leen.
- `uv run poe report` tras cambiar `docs/informe_final.md`, con Docker encendido, y commitea el PDF.
- `uv run poe requirements` tras cambiar las dependencias.

## Autoría y publicación

- Autor único: Joaquín Castro Salas. Commits en español con prefijo convencional y sin línea de coautoría ni atribución.
- Todo es local, sin hosting remoto. El push, los PR y los merges los decide el autor.
