#!/bin/bash

# Configuration
IMAGE_NAME="mlops-frontend"
VERSION="latest"
DOCKER_USER="votre-username" # À modifier

# Build
docker build -t $DOCKER_USER/$IMAGE_NAME:$VERSION .

# Push
# docker push $DOCKER_USER/$IMAGE_NAME:$VERSION

echo "Frontend image built successfully!"
