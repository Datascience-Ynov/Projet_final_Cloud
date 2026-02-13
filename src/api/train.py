import os
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server-mlops-sadiya-mourad.francecentral.azurecontainer.io:5000/")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "Fashion_MNIST_Training")
DATA_PATH = os.getenv("DATA_PATH", "data/fashion_mnist_agnostic.npz")


def load_data(data_path: str):
    data = np.load(data_path)
    return data["X_train"], data["y_train"], data["X_test"], data["y_test"]


def train_model(model, model_name, X_train, y_train, X_test, y_test):
    """Train a single model and log to MLflow."""
    print(f"\n{'='*60}")
    print(f"Training {model_name}...")
    print(f"{'='*60}")
    
    with mlflow.start_run(run_name=f"{model_name}_Training"):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        mlflow.log_param("model_type", model_name)
        mlflow.log_param("model_params", str(model.get_params()))
        mlflow.log_metric("accuracy", acc)

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=f"fashion-mnist-{model_name.lower()}",
        )

        print(f"✓ {model_name} logged to MLflow with accuracy={acc:.4f}")
        return acc


def train_all_models():
    """Train multiple models and log them to MLflow."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, y_train, X_test, y_test = load_data(DATA_PATH)
    
    print(f"Data loaded: {len(X_train)} training samples, {len(X_test)} test samples")

    # Define models to train
    models = {
        "SVC": SVC(C=1.0, kernel="rbf", max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=20, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=5, random_state=42, eval_metric='mlogloss')
    }

    results = {}
    for model_name, model in models.items():
        try:
            acc = train_model(model, model_name, X_train, y_train, X_test, y_test)
            results[model_name] = acc
        except Exception as e:
            print(f"✗ Error training {model_name}: {e}")
            results[model_name] = None
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for model_name, acc in results.items():
        if acc is not None:
            print(f"{model_name:20s}: {acc:.4f}")
        else:
            print(f"{model_name:20s}: FAILED")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    train_all_models()
