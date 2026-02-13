#!/bin/bash

# Configuration
PROJECT_ID="ynov-486913"
SERVICE_NAME="mlops-backend-api"
REGION="europe-west1"
IMAGE_NAME="mlops-backend"
MLFLOW_URL="http://mlflow-server-mlops-sadiya-mourad.francecentral.azurecontainer.io:5000"

echo "=== Déploiement Backend FastAPI sur Google Cloud Run (Build Local) ==="

# 1. Vérification Docker
echo -e "\n1. Vérification de Docker..."
if ! command -v docker &> /dev/null; then
    echo "❌ Docker n'est pas installé"
    exit 1
fi

# 2. Configuration du projet
echo -e "\n2. Configuration du projet GCP..."
gcloud config set project $PROJECT_ID

# 3. Activation des APIs
echo -e "\n3. Activation des APIs..."
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# 4. Configuration Docker pour GCR
echo -e "\n4. Configuration Docker pour Google Container Registry..."
gcloud auth configure-docker

# 5. Build de l'image localement
echo -e "\n5. Construction de l'image Docker localement..."
docker build -t gcr.io/$PROJECT_ID/$IMAGE_NAME:latest .

if [ $? -ne 0 ]; then
    echo "❌ Erreur lors du build Docker"
    exit 1
fi

# 6. Push de l'image vers GCR
echo -e "\n6. Push de l'image vers Google Container Registry..."
docker push gcr.io/$PROJECT_ID/$IMAGE_NAME:latest

if [ $? -ne 0 ]; then
    echo "❌ Erreur lors du push vers GCR"
    exit 1
fi

# 7. Déploiement sur Cloud Run
echo -e "\n7. Déploiement sur Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image gcr.io/$PROJECT_ID/$IMAGE_NAME:latest \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --set-env-vars MLFLOW_TRACKING_URI=$MLFLOW_URL \
  --set-env-vars MLFLOW_MODEL_URI=models:/fashion-mnist-sklearn/Production \
  --memory 2Gi \
  --cpu 1 \
  --min-instances 0 \
  --max-instances 10 \
  --port 8000

# 8. Récupération de l'URL
echo -e "\n8. Récupération de l'URL du service..."
BACKEND_URL=$(gcloud run services describe $SERVICE_NAME --platform managed --region $REGION --format 'value(status.url)')

if [ -z "$BACKEND_URL" ]; then
    echo "❌ Erreur: Impossible de récupérer l'URL"
    exit 1
fi

echo -e "\n✅ Déploiement réussi!"
echo "=========================================="
echo "Backend FastAPI est accessible sur:"
echo "$BACKEND_URL"
echo "=========================================="
echo ""
echo "Pour configurer le frontend Heroku:"
echo "heroku config:set BACKEND_URL=$BACKEND_URL --app votre-app-frontend"
echo "heroku config:set MLFLOW_URL=$MLFLOW_URL --app votre-app-frontend"
