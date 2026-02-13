import os
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "Fashion_MNIST_Training")
MODEL_NAME = os.getenv("MODEL_NAME", "fashion-mnist-sklearn")
DATA_PATH = os.getenv("DATA_PATH", "data/fashion_mnist_agnostic.npz")


def load_data(data_path: str):
    data = np.load(data_path)
    return data["X_train"], data["y_train"], data["X_test"], data["y_test"]


def train_and_log():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, y_train, X_test, y_test = load_data(DATA_PATH)

    model = SVC(C=1.0, kernel="rbf")
    with mlflow.start_run(run_name="SVC_Training"):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        mlflow.log_param("model", "SVC")
        mlflow.log_metric("accuracy", acc)

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )

        print(f"Model logged to MLflow with accuracy={acc:.4f}")


if __name__ == "__main__":
    train_and_log()
