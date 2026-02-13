FROM python:3.11-slim

WORKDIR /app

# Install MLflow and dependencies
RUN pip install --no-cache-dir mlflow==2.16.0 psycopg2-binary azure-storage-blob

# Create directories for artifacts
RUN mkdir -p /mlflow/artifacts /mlflow/backend

# Expose MLflow port
EXPOSE 5000

# Set default backend store (can be overridden with env vars)
ENV BACKEND_STORE_URI=sqlite:///mlflow/backend/mlflow.db
ENV ARTIFACT_ROOT=/mlflow/artifacts
ENV HOST=0.0.0.0
ENV PORT=5000

# Start MLflow server
CMD mlflow server \
    --backend-store-uri ${BACKEND_STORE_URI} \
    --default-artifact-root ${ARTIFACT_ROOT} \
    --host ${HOST} \
    --port ${PORT}
