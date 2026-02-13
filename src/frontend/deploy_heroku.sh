#!/bin/bash

# Configuration
APP_NAME="mlops-fashion-mnist-frontend"  # Nom de votre app Heroku (unique)
BACKEND_URL="https://mlops-backend-api-2ahwjpnqfq-ew.a.run.app"  # À configurer après déploiement GCP
MLFLOW_URL="http://mlflow-server-mlops-sadiya-mourad.francecentral.azurecontainer.io:5000"

echo "=== Déploiement Frontend Streamlit sur Heroku ==="

# 1. Vérification Heroku CLI
echo -e "\n1. Vérification de Heroku CLI..."
if ! command -v heroku &> /dev/null; then
    echo "❌ Heroku CLI n'est pas installé"
    echo "Installez-le: curl https://cli-assets.heroku.com/install.sh | sh"
    exit 1
fi

# 2. Connexion Heroku
echo -e "\n2. Connexion à Heroku..."
heroku auth:whoami > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Connexion à Heroku requise..."
    heroku login
fi

# 3. Connexion au Container Registry
echo -e "\n3. Connexion au Container Registry Heroku..."
heroku container:login

# 4. Création de l'application (si n'existe pas)
echo -e "\n4. Vérification/création de l'application..."
heroku apps:info --app $APP_NAME > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Création de l'application $APP_NAME..."
    heroku create $APP_NAME --region eu
    if [ $? -ne 0 ]; then
        echo "❌ Erreur lors de la création de l'app"
        echo "Le nom '$APP_NAME' est peut-être déjà pris. Modifiez APP_NAME dans le script."
        exit 1
    fi
fi

# 4.5 Configuration du stack container
echo -e "\n4.5. Configuration du stack container..."
heroku stack:set container --app $APP_NAME
if [ $? -ne 0 ]; then
    echo "❌ Erreur lors de la configuration du stack"
    exit 1
fi

# 5. Demander l'URL du backend GCP si pas configurée
if [ -z "$BACKEND_URL" ]; then
    echo -e "\n⚠️  BACKEND_URL non configurée"
    echo "Récupérez l'URL de votre backend GCP Cloud Run et relancez le script avec:"
    echo "BACKEND_URL='https://mlops-backend-api-2ahwjpnqfq-ew.a.run.app/' ./deploy_heroku.sh"
    echo ""
    echo "Ou modifiez la variable BACKEND_URL dans le script deploy_heroku.sh"
    exit 1
fi

# 6. Construction et push de l'image Docker
echo -e "\n5. Construction et push de l'image Docker vers Heroku..."
heroku container:push web --app $APP_NAME

if [ $? -ne 0 ]; then
    echo "❌ Erreur lors du build/push Docker"
    exit 1
fi

# 7. Release de l'application
echo -e "\n6. Release de l'application..."
heroku container:release web --app $APP_NAME

# 8. Configuration des variables d'environnement
echo -e "\n7. Configuration des variables d'environnement..."
heroku config:set BACKEND_URL=$BACKEND_URL --app $APP_NAME
heroku config:set MLFLOW_URL=$MLFLOW_URL --app $APP_NAME

# 9. Récupération de l'URL
FRONTEND_URL=$(heroku apps:info --app $APP_NAME --json | python3 -c "import sys, json; print(json.load(sys.stdin)['app']['web_url'])")

echo -e "\n✅ Déploiement réussi!"
echo "=========================================="
echo "Frontend Streamlit est accessible sur:"
echo "$FRONTEND_URL"
echo "=========================================="
echo ""
echo "Commandes utiles:"
echo "- Voir les logs: heroku logs --tail --app $APP_NAME"
echo "- Ouvrir l'app: heroku open --app $APP_NAME"
echo "- Redémarrer: heroku restart --app $APP_NAME"
