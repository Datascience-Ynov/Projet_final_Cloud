# MLflow Deployment on Azure


```bash
cd mlflow_deploy
chmod +x deploy_azure.sh

# Modifier les variables dans deploy_azure.sh
# Décommenter les commandes et exécuter
./deploy_azure.sh
```

## Configuration requise

1. **Azure CLI installé**
   ```bash
   # Ubuntu/Debian
   curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
   ```

2. **Connexion Azure**
   ```bash
   az login
   ```

## Déploiement pas à pas

### Azure Container Instances (ACI) - Développement/Test

```bash
# Variables
RESOURCE_GROUP="mlops-rg"
APP_NAME="mlflow-server-mlops"
LOCATION="francecentral"

# Création du groupe de ressources
az group create --name $RESOURCE_GROUP --location $LOCATION

# Build et déploiement direct (sans ACR)
az container create \
  --resource-group $RESOURCE_GROUP \
  --name $APP_NAME \
  --image ghcr.io/mlflow/mlflow:latest \
  --cpu 1 \
  --memory 2 \
  --dns-name-label $APP_NAME \
  --ports 5000 \
  --command-line "mlflow server --host 0.0.0.0 --port 5000"

# Récupération de l'URL
az container show --resource-group $RESOURCE_GROUP --name $APP_NAME --query ipAddress.fqdn -o tsv
```



## Après déploiement

1. **Récupérer l'URL MLflow**
   ```bash
   # Pour ACI
   MLFLOW_URL=$(az container show --resource-group $RESOURCE_GROUP --name $APP_NAME --query ipAddress.fqdn -o tsv)
   echo "http://${MLFLOW_URL}:5000"

   # Pour App Service
   echo "https://${APP_NAME}.azurewebsites.net"
   ```

2. **Configurer dans votre backend**
   ```bash
   # Mettre à jour MLFLOW_TRACKING_URI dans src/api/main.py
   export MLFLOW_TRACKING_URI=http://votre-url-mlflow:5000
   ```

3. **Tester la connexion**
   ```python
   import mlflow
   mlflow.set_tracking_uri("http://votre-url-mlflow:5000")
   print(mlflow.get_tracking_uri())
   ```

