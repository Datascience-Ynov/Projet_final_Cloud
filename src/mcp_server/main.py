import mcp.server.fastmcp as fastmcp
import pickle
import io
import optuna
import os
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import mlflow

try:
    import torch
except Exception:
    torch = None

try:
    import tensorflow as tf
except Exception:
    tf = None

mcp_server = fastmcp.FastMCP("MLOps Optimization Server")

@mcp_server.tool()
async def detect_model_framework(model_path: str) -> str:
    """Detects the framework of a given model file path."""
    if not os.path.exists(model_path):
        return "error: file not found"
    
    # Try PyTorch
    if torch is not None:
        try:
            torch.load(model_path, weights_only=True)
            return "pytorch"
        except:
            pass

    # Try Sklearn (pickle)
    try:
        with open(model_path, "rb") as f:
            pickle.load(f)
        return "sklearn"
    except:
        pass

    # Try Keras/TF
    if tf is not None:
        try:
            tf.keras.models.load_model(model_path)
            return "tensorflow"
        except:
            pass

    return "unknown"

def load_data(data_path: str):
    """Utility to load data from a .npz file."""
    data = np.load(data_path)
    return data['X_train'], data['y_train'], data['X_test'], data['y_test']

@mcp_server.tool()
def grid_search_optimizer(model_name: str, search_space: dict, data_path: str, experiment_name: str):
    """Runs GridSearchCV and logs to MLflow."""
    X_train, y_train, _, _ = load_data(data_path)
    
    models = {
        "svc": SVC(),
        "random_forest": RandomForestClassifier(),
        "mlp": MLPClassifier(),
        "xgboost": XGBClassifier()
    }
    
    model = models.get(model_name.lower())
    if not model:
        return {"error": f"Model {model_name} not supported"}

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"GridSearch_{model_name}"):
        search = GridSearchCV(model, search_space, cv=3)
        search.fit(X_train, y_train)
        
        mlflow.log_params(search.best_params_)
        mlflow.log_metric("best_score", search.best_score_)
        
        return {
            "best_params": search.best_params_,
            "best_score": search.best_score_,
            "run_id": mlflow.active_run().info.run_id
        }

@mcp_server.tool()
def random_search_optimizer(model_name: str, search_space: dict, data_path: str, experiment_name: str, n_iter: int = 10):
    """Runs RandomizedSearchCV and logs to MLflow."""
    X_train, y_train, _, _ = load_data(data_path)
    
    models = {
        "svc": SVC(),
        "random_forest": RandomForestClassifier(),
        "mlp": MLPClassifier(),
        "xgboost": XGBClassifier()
    }
    
    model = models.get(model_name.lower())
    if not model:
        return {"error": f"Model {model_name} not supported"}

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"RandomSearch_{model_name}"):
        search = RandomizedSearchCV(model, search_space, n_iter=n_iter, cv=3)
        search.fit(X_train, y_train)
        
        mlflow.log_params(search.best_params_)
        mlflow.log_metric("best_score", search.best_score_)
        
        return {
            "best_params": search.best_params_,
            "best_score": search.best_score_,
            "run_id": mlflow.active_run().info.run_id
        }

@mcp_server.tool()
def bayesian_optimizer(model_name: str, search_space: dict, data_path: str, experiment_name: str, n_trials: int = 10):
    """Runs Bayesian optimization with Optuna and logs to MLflow."""
    X_train, y_train, _, _ = load_data(data_path)
    
    mlflow.set_experiment(experiment_name)
    
    def objective(trial):
        with mlflow.start_run(run_name=f"Bayesian_Trial_{trial.number}", nested=True):
            params = {}
            for k, v in search_space.items():
                if isinstance(v, list):
                    params[k] = trial.suggest_categorical(k, v)
                elif isinstance(v, dict) and "low" in v and "high" in v:
                    if v.get("log", False):
                        params[k] = trial.suggest_float(k, v["low"], v["high"], log=True)
                    else:
                        params[k] = trial.suggest_float(k, v["low"], v["high"])
            
            # Simple instantiation for demo
            if model_name.lower() == "svc":
                model = SVC(**params)
            elif model_name.lower() == "random_forest":
                model = RandomForestClassifier(**params)
            elif model_name.lower() == "mlp":
                model = MLPClassifier(**params)
            else:
                return 0
            
            from sklearn.model_selection import cross_val_score
            score = cross_val_score(model, X_train, y_train, cv=3).mean()
            
            mlflow.log_params(params)
            mlflow.log_metric("accuracy", score)
            return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    with mlflow.start_run(run_name=f"Bayesian_Best_{model_name}"):
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_accuracy", study.best_value)

    return {
        "best_params": study.best_params,
        "best_score": study.best_value,
        "study_name": study.study_name
    }

if __name__ == "__main__":
    mcp_server.run()
