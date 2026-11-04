# API and interface images, without TensorFlow, MLflow or SHAP: only the base and `ui` dependencies.
# The champion is a Random Forest, so XGBoost's GPU library (over 300 MB) is left out of both.

FROM python:3.12.15-slim-bookworm AS build
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1 \
    UV_PYTHON_DOWNLOADS=0
WORKDIR /app
COPY pyproject.toml uv.lock README.md .python-version ./

FROM build AS build-api
RUN uv sync --locked --no-install-project --no-install-package nvidia-nccl-cu13
COPY src/ src/
RUN uv sync --locked --no-editable --no-install-package nvidia-nccl-cu13

FROM build AS build-ui
RUN uv sync --locked --no-install-project --no-install-package nvidia-nccl-cu13 --extra ui
COPY src/ src/
RUN uv sync --locked --no-editable --no-install-package nvidia-nccl-cu13 --extra ui

FROM python:3.12.15-slim-bookworm AS runtime
RUN useradd --create-home --uid 10001 app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOTEL_ML_MODELS_DIR=/app/models \
    HOTEL_ML_OUTPUTS_DIR=/app/outputs
WORKDIR /app
USER app

FROM runtime AS api
COPY --from=build-api /app/.venv .venv
COPY models/champion/ models/champion/
COPY outputs/final_evaluation/ outputs/final_evaluation/
EXPOSE 8000
CMD ["uvicorn", "hotel_cancellation.serving.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM runtime AS ui
COPY --from=build-ui /app/.venv .venv
COPY models/champion/manifest.json models/champion/manifest.json
COPY outputs/ outputs/
COPY .streamlit/ .streamlit/
EXPOSE 8501
CMD ["streamlit", "run", ".venv/lib/python3.12/site-packages/hotel_cancellation/serving/ui/app.py", \
     "--server.address", "0.0.0.0", "--server.port", "8501", "--server.headless", "true", "--server.fileWatcherType", "none"]
