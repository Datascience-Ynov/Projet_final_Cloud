import anyio
import json
import logging
from mcp.server import Server, stdio_server
import mcp.types as types
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

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("mcp_server")

server = Server("MLOps Optimization Server")

async def detect_model_framework(model_path: str) -> str:
    """Detects the framework of a given model file path."""
    logger.info("detect_model_framework called with model_path=%s", model_path)
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

def grid_search_optimizer(model_name: str, search_space: dict, data_path: str, experiment_name: str):
    """Runs GridSearchCV and logs to MLflow."""
    logger.info(
        "grid_search_optimizer called with model_name=%s data_path=%s experiment_name=%s search_space=%s",
        model_name,
        data_path,
        experiment_name,
        search_space,
    )
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

def random_search_optimizer(model_name: str, search_space: dict, data_path: str, experiment_name: str, n_iter: int = 10):
    """Runs RandomizedSearchCV and logs to MLflow."""
    logger.info(
        "random_search_optimizer called with model_name=%s data_path=%s experiment_name=%s n_iter=%s search_space=%s",
        model_name,
        data_path,
        experiment_name,
        n_iter,
        search_space,
    )
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

def bayesian_optimizer(model_name: str, search_space: dict, data_path: str, experiment_name: str, n_trials: int = 10):
    """Runs Bayesian optimization with Optuna and logs to MLflow."""
    logger.info(
        "bayesian_optimizer called with model_name=%s data_path=%s experiment_name=%s n_trials=%s search_space=%s",
        model_name,
        data_path,
        experiment_name,
        n_trials,
        search_space,
    )
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

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="detect_model_framework",
            description="Detect the framework of a model file.",
            inputSchema={
                "type": "object",
                "properties": {"model_path": {"type": "string"}},
                "required": ["model_path"],
            },
        ),
        types.Tool(
            name="grid_search_optimizer",
            description="Run GridSearchCV and log to MLflow.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_name": {"type": "string"},
                    "search_space": {"type": "object"},
                    "data_path": {"type": "string"},
                    "experiment_name": {"type": "string"},
                },
                "required": ["model_name", "search_space", "data_path", "experiment_name"],
            },
        ),
        types.Tool(
            name="random_search_optimizer",
            description="Run RandomizedSearchCV and log to MLflow.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_name": {"type": "string"},
                    "search_space": {"type": "object"},
                    "data_path": {"type": "string"},
                    "experiment_name": {"type": "string"},
                    "n_iter": {"type": "integer", "minimum": 1},
                },
                "required": ["model_name", "search_space", "data_path", "experiment_name"],
            },
        ),
        types.Tool(
            name="bayesian_optimizer",
            description="Run Bayesian optimization with Optuna and log to MLflow.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_name": {"type": "string"},
                    "search_space": {"type": "object"},
                    "data_path": {"type": "string"},
                    "experiment_name": {"type": "string"},
                    "n_trials": {"type": "integer", "minimum": 1},
                },
                "required": ["model_name", "search_space", "data_path", "experiment_name"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    logger.info("MCP call_tool: %s arguments=%s", name, arguments)
    if name == "detect_model_framework":
        result = await detect_model_framework(arguments["model_path"])
        return [types.TextContent(type="text", text=result)]

    if name == "grid_search_optimizer":
        result = grid_search_optimizer(
            arguments["model_name"],
            arguments["search_space"],
            arguments["data_path"],
            arguments["experiment_name"],
        )
        return [types.TextContent(type="text", text=json.dumps(result))]

    if name == "random_search_optimizer":
        result = random_search_optimizer(
            arguments["model_name"],
            arguments["search_space"],
            arguments["data_path"],
            arguments["experiment_name"],
            n_iter=arguments.get("n_iter", 10),
        )
        return [types.TextContent(type="text", text=json.dumps(result))]

    if name == "bayesian_optimizer":
        result = bayesian_optimizer(
            arguments["model_name"],
            arguments["search_space"],
            arguments["data_path"],
            arguments["experiment_name"],
            n_trials=arguments.get("n_trials", 10),
        )
        return [types.TextContent(type="text", text=json.dumps(result))]

    return [types.TextContent(type="text", text=f"Unknown tool: {name}")]


async def _main() -> None:
    logger.info("MCP server starting")
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    anyio.run(_main)
