FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./api

ENV MLFLOW_TRACKING_URI="http://mlflow-server-mlops-sadiya-mourad.francecentral.azurecontainer.io:5000/"
ENV MLFLOW_MODEL_URI=models:/fashion-mnist-sklearn/Production

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
