from fastapi import FastAPI, UploadFile, File
import os
import uuid
import json
import mlflow
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import optuna

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_MODEL_URI = os.getenv("MLFLOW_MODEL_URI", "models:/fashion-mnist-sklearn/Production")

app = FastAPI(title="MLOps Backend")

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

_model_cache: Optional[mlflow.pyfunc.PyFuncModel] = None

@app.get("/")
def read_root():
    return {"message": "MLOps Backend is running"}

@app.post("/upload-data")
async def upload_data(file: UploadFile = File(...)):
    """Uploads a dataset (.npz format for simplicity)."""
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}.npz")
    
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)
        
    return {"file_id": file_id, "file_path": file_path}

@app.post("/upload-model")
async def upload_model(file: UploadFile = File(...)):
    """Uploads a model file."""
    file_id = str(uuid.uuid4())
    ext = file.filename.split('.')[-1]
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}.{ext}")
    
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)
        
    return {"model_id": file_id, "model_path": file_path}

def load_data(data_path: str):
    """Load data from .npz file."""
    data = np.load(data_path)
    return data['X_train'], data['y_train'], data['X_test'], data['y_test']

@app.post("/optimize")
async def start_optimization(config: Dict[str, Any]):
    """
    Run hyperparameter optimization directly.
    """
    try:
        model_name = config["model_name"].lower()
        strategy = config.get("strategy", "grid")
        search_space = config["search_space"]
        data_path = os.path.abspath(config["data_path"])
        experiment_name = config["experiment_name"]
        
        X_train, y_train, _, _ = load_data(data_path)
        
        models = {
            "svc": SVC(),
            "random_forest": RandomForestClassifier(),
            "mlp": MLPClassifier(),
            "xgboost": XGBClassifier()
        }
        
        model = models.get(model_name)
        if not model:
            return {"status": "error", "message": f"Model {model_name} not supported"}
        
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_name)
        
        if strategy == "grid":
            with mlflow.start_run(run_name=f"GridSearch_{model_name}"):
                search = GridSearchCV(model, search_space, cv=3)
                search.fit(X_train, y_train)
                
                mlflow.log_params(search.best_params_)
                mlflow.log_metric("best_score", search.best_score_)
                
                active_run = mlflow.active_run()
                result = {
                    "best_params": search.best_params_,
                    "best_score": search.best_score_,
                    "run_id": active_run.info.run_id,
                    "experiment_id": active_run.info.experiment_id
                }
        
        elif strategy == "random":
            n_iter = config.get("n_iter", 10)
            with mlflow.start_run(run_name=f"RandomSearch_{model_name}"):
                search = RandomizedSearchCV(model, search_space, n_iter=n_iter, cv=3)
                search.fit(X_train, y_train)
                
                mlflow.log_params(search.best_params_)
                mlflow.log_metric("best_score", search.best_score_)
                
                active_run = mlflow.active_run()
                result = {
                    "best_params": search.best_params_,
                    "best_score": search.best_score_,
                    "run_id": active_run.info.run_id,
                    "experiment_id": active_run.info.experiment_id
                }
        
        elif strategy == "bayesian":
            n_trials = config.get("n_trials", 10)
            
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
                    
                    if model_name == "svc":
                        m = SVC(**params)
                    elif model_name == "random_forest":
                        m = RandomForestClassifier(**params)
                    elif model_name == "mlp":
                        m = MLPClassifier(**params)
                    else:
                        return 0
                    
                    from sklearn.model_selection import cross_val_score
                    score = cross_val_score(m, X_train, y_train, cv=3).mean()
                    
                    mlflow.log_params(params)
                    mlflow.log_metric("accuracy", score)
                    return score
            
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials)
            
            with mlflow.start_run(run_name=f"Bayesian_Best_{model_name}"):
                mlflow.log_params(study.best_params)
                mlflow.log_metric("best_accuracy", study.best_value)
                active_run = mlflow.active_run()
            
            result = {
                "best_params": study.best_params,
                "best_score": study.best_value,
                "study_name": study.study_name,
                "run_id": active_run.info.run_id,
                "experiment_id": active_run.info.experiment_id
            }
        else:
            return {"status": "error", "message": f"Unknown strategy: {strategy}"}
        
        return {"status": "success", "result": result}
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/mlflow-info")
def get_mlflow_info():
    """Returns MLflow tracking URI."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return {"tracking_uri": mlflow.get_tracking_uri(), "model_uri": MLFLOW_MODEL_URI}

@app.get("/models")
def list_models():
    """List all registered models in MLflow with their versions and stages."""
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.MlflowClient()
        
        registered_models = client.search_registered_models()
        models_info = []
        
        for rm in registered_models:
            model_versions = []
            for mv in rm.latest_versions:
                model_versions.append({
                    "version": mv.version,
                    "stage": mv.current_stage,
                    "uri": f"models:/{rm.name}/{mv.version}",
                    "stage_uri": f"models:/{rm.name}/{mv.current_stage}" if mv.current_stage != "None" else None
                })
            
            models_info.append({
                "name": rm.name,
                "versions": model_versions
            })
        
        return {"status": "success", "models": models_info}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/predict")
def predict(payload: Dict[str, Any]):
    """
    Expects JSON: {"features": [[...], [...]]} or {"features": [...]}
    Optional: {"model_uri": "models:/fashion-mnist-sklearn/Production"}
    Returns model predictions from MLflow model registry.
    """
    global _model_cache

    if "features" not in payload:
        return {"status": "error", "message": "Missing 'features' in payload"}

    features = payload["features"]
    if not isinstance(features, list):
        return {"status": "error", "message": "'features' must be a list"}

    if len(features) == 0:
        return {"status": "error", "message": "'features' list is empty"}

    if not isinstance(features[0], list):
        features = [features]

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Allow dynamic model selection
    model_uri = payload.get("model_uri", MLFLOW_MODEL_URI)
    
    # Reload model if URI changed or not cached
    if _model_cache is None or payload.get("model_uri"):
        _model_cache = mlflow.pyfunc.load_model(model_uri)

    df = pd.DataFrame(features)
    preds = _model_cache.predict(df)
    preds_list = np.asarray(preds).tolist()
    return {"status": "success", "predictions": preds_list, "model_used": model_uri}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
