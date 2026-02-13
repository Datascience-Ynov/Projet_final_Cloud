#!/bin/bash

# Configuration
IMAGE_NAME="mlops-backend"
VERSION="latest"
DOCKER_USER="votre-username" # À modifier

# Build from repo root
docker build -t $DOCKER_USER/$IMAGE_NAME:$VERSION -f src/api/Dockerfile .

# Push to Docker Hub (uncomment when ready)
# docker push $DOCKER_USER/$IMAGE_NAME:$VERSION

echo "Backend image built successfully!"
echo "To push: docker push $DOCKER_USER/$IMAGE_NAME:$VERSION"
