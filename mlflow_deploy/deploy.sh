#!/bin/bash

# Configuration
APP_NAME="mlflow-server-mlops-sadiya-mourad"
RESOURCE_GROUP="mlops-rg"
LOCATION="francecentral"

echo "=== Déploiement MLflow sur Azure Container Instances (Simple) ==="

# 1. Connexion à Azure (si pas déjà connecté)
echo -e "\n1. Vérification de la connexion Azure..."
az account show > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Connexion à Azure requise..."
    az login
fi

# 2. Création du groupe de ressources
echo -e "\n2. Création du groupe de ressources..."
az group create --name $RESOURCE_GROUP --location $LOCATION

# 3. Enregistrement du provider Container Instance
echo -e "\n3. Enregistrement du provider Azure Container Instance..."
az provider register --namespace Microsoft.ContainerInstance
echo "Attente de l'enregistrement (peut prendre 1-2 minutes)..."
az provider show --namespace Microsoft.ContainerInstance --query "registrationState" -o tsv
while [ "$(az provider show --namespace Microsoft.ContainerInstance --query registrationState -o tsv)" != "Registered" ]; do
    echo -n "."
    sleep 5
done
echo -e "\n✓ Provider enregistré!"

# 4. Déploiement direct avec image publique MLflow
echo -e "\n4. Déploiement du conteneur MLflow..."
az container create \
  --resource-group $RESOURCE_GROUP \
  --name $APP_NAME \
  --image ghcr.io/mlflow/mlflow:latest \
  --os-type Linux \
  --cpu 1 \
  --memory 2 \
  --dns-name-label $APP_NAME \
  --ports 5000 \
  --command-line "mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root /mlflow/artifacts --allowed-hosts '*'"

# 5. Attendre que le conteneur soit prêt
echo -e "\n5. Attente du démarrage du conteneur..."
sleep 10

# 6. Récupération de l'URL publique
echo -e "\n6. Récupération de l'URL MLflow..."
MLFLOW_FQDN=$(az container show --resource-group $RESOURCE_GROUP --name $APP_NAME --query ipAddress.fqdn -o tsv)

if [ -z "$MLFLOW_FQDN" ]; then
    echo "❌ Erreur: Impossible de récupérer l'URL"
    exit 1
fi

MLFLOW_URL="http://${MLFLOW_FQDN}:5000"

echo -e "\n✅ Déploiement réussi!"
echo "=========================================="
echo "MLflow est accessible sur:"
echo "  ${MLFLOW_URL}"
echo "=========================================="
echo ""
echo "Configuration pour votre backend:"
echo "  export MLFLOW_TRACKING_URI=${MLFLOW_URL}"
echo ""
echo "Configuration Heroku (frontend):"
echo "  heroku config:set MLFLOW_URL=${MLFLOW_URL} --app votre-app-frontend"
echo ""
echo "Commandes utiles:"
echo "  - Voir les logs: az container logs --resource-group $RESOURCE_GROUP --name $APP_NAME --follow"
echo "  - Redémarrer: az container restart --resource-group $RESOURCE_GROUP --name $APP_NAME"
echo "  - Supprimer: az group delete --name $RESOURCE_GROUP --yes --no-wait"
echo ""

# 7. Afficher les logs initiaux
echo -e "\n7. Logs du conteneur (Ctrl+C pour quitter):"
az container logs --resource-group $RESOURCE_GROUP --name $APP_NAME --follow
