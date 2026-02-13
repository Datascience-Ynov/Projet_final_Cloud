# Déploiement Backend FastAPI sur Google Cloud Run

## Prérequis

1. **Google Cloud SDK installé**
   ```bash
   # Ubuntu/Debian
   curl https://sdk.cloud.google.com | bash
   exec -l $SHELL
   gcloud init
   ```

2. **Projet GCP créé**
   - Créer un projet sur https://console.cloud.google.com
   - Noter le PROJECT_ID

3. **Facturation activée**
   - Cloud Run nécessite la facturation (offre gratuite disponible)

## Déploiement rapide

```bash
cd src/api

# Éditer deploy_gcp.sh et modifier:
# - PROJECT_ID="votre-project-id-gcp"
# - Vérifier MLFLOW_URL (déjà configuré avec l'URL Azure)

# Déployer
./deploy_gcp.sh
```

## Déploiement manuel pas à pas

### 1. Connexion et configuration

```bash
gcloud auth login
gcloud config set project VOTRE_PROJECT_ID
```

### 2. Activation des APIs

```bash
gcloud services enable run.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable cloudbuild.googleapis.com
```

### 3. Build de l'image (depuis la racine du projet)

```bash
cd /path/to/Fashion_MNIST_AI

gcloud builds submit --tag gcr.io/VOTRE_PROJECT_ID/mlops-backend:latest src/api
```

### 4. Déploiement sur Cloud Run

```bash
gcloud run deploy mlops-backend-api \
  --image gcr.io/VOTRE_PROJECT_ID/mlops-backend:latest \
  --platform managed \
  --region europe-west1 \
  --allow-unauthenticated \
  --set-env-vars MLFLOW_TRACKING_URI=http://mlflow-server-mlops-sadiya-mourad.francecentral.azurecontainer.io:5000 \
  --set-env-vars MLFLOW_MODEL_URI=models:/fashion-mnist-sklearn/Production \
  --memory 2Gi \
  --cpu 1 \
  --port 8000
```

### 5. Récupération de l'URL

```bash
gcloud run services describe mlops-backend-api \
  --platform managed \
  --region europe-west1 \
  --format 'value(status.url)'
```

## Configuration du Dockerfile pour Cloud Run

Le Dockerfile actuel (`src/api/Dockerfile`) est déjà compatible avec Cloud Run car:
- ✅ Expose le port 8000
- ✅ Utilise des variables d'environnement
- ✅ Image Linux compatible

Cloud Run définit automatiquement la variable `PORT` mais notre backend utilise le port 8000 par défaut.

## Variables d'environnement

Cloud Run injecte automatiquement:
- `MLFLOW_TRACKING_URI`: URL MLflow sur Azure
- `MLFLOW_MODEL_URI`: URI du modèle MLflow

## Test de l'API déployée

```bash
BACKEND_URL="https://mlops-backend-api-xxx-ew.a.run.app"

# Test endpoint root
curl $BACKEND_URL/

# Test liste des modèles MLflow
curl $BACKEND_URL/models

# Test prédiction
curl -X POST $BACKEND_URL/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [[0.1, 0.2, 0.3]], "model_uri": "models:/fashion-mnist-sklearn/Production"}'
```

## Commandes utiles

```bash
# Voir les logs
gcloud run services logs read mlops-backend-api --region europe-west1 --limit 50

# Mettre à jour les variables d'environnement
gcloud run services update mlops-backend-api \
  --region europe-west1 \
  --set-env-vars MLFLOW_TRACKING_URI=nouvelle-url

# Scaler (min/max instances)
gcloud run services update mlops-backend-api \
  --region europe-west1 \
  --min-instances 0 \
  --max-instances 10

# Supprimer le service
gcloud run services delete mlops-backend-api --region europe-west1
```

## Configuration frontend Heroku

Une fois le backend déployé, configurer le frontend:

```bash
heroku config:set BACKEND_URL=https://mlops-backend-api-xxx-ew.a.run.app --app votre-app-frontend
heroku config:set MLFLOW_URL=http://mlflow-server-mlops-sadiya-mourad.francecentral.azurecontainer.io:5000 --app votre-app-frontend
```

## Coûts estimés (GCP)

Cloud Run (gratuit jusqu'à):
- 2 millions de requêtes/mois
- 360,000 GB-secondes de mémoire/mois
- 180,000 vCPU-secondes/mois

Au-delà: ~0.024€/heure d'utilisation

## Architecture finale

```
Frontend (Heroku) → Backend (GCP Cloud Run) → MLflow (Azure ACI)
       ↓                    ↓                        ↓
   Streamlit           FastAPI                  Tracking Server
```

## Troubleshooting

### Erreur "Permission denied"
```bash
gcloud auth application-default login
```

### Erreur "billing not enabled"
- Aller sur https://console.cloud.google.com/billing
- Activer la facturation pour le projet

### Logs d'erreur
```bash
gcloud run services logs read mlops-backend-api --region europe-west1 --limit 100
```
