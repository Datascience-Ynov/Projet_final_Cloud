from fastapi import FastAPI, UploadFile, File
import os
import uuid
import json
import mlflow
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import sys

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

@app.post("/optimize")
async def start_optimization(config: Dict[str, Any]):
    """
    Triggers optimization via MCP.
    """
    server_script = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "mcp_server", "main.py")
    )
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[server_script],
        env=os.environ.copy()
    )

    strategy = config.get("strategy", "grid")
    tool_name = f"{strategy}_search_optimizer"
    if strategy == "bayesian":
        tool_name = "bayesian_optimizer"

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # Call the MCP tool
                result = await session.call_tool(
                    tool_name,
                    arguments={
                        "model_name": config["model_name"],
                        "search_space": config["search_space"],
                        "data_path": os.path.abspath(config["data_path"]),
                        "experiment_name": config["experiment_name"]
                    }
                )
                
                return {"status": "success", "result": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/mlflow-info")
def get_mlflow_info():
    """Returns MLflow tracking URI."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return {"tracking_uri": mlflow.get_tracking_uri(), "model_uri": MLFLOW_MODEL_URI}

@app.post("/predict")
def predict(payload: Dict[str, Any]):
    """
    Expects JSON: {"features": [[...], [...]]} or {"features": [...]}
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
    if _model_cache is None:
        _model_cache = mlflow.pyfunc.load_model(MLFLOW_MODEL_URI)

    df = pd.DataFrame(features)
    preds = _model_cache.predict(df)
    preds_list = np.asarray(preds).tolist()
    return {"status": "success", "predictions": preds_list}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
