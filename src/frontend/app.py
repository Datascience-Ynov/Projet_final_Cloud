import streamlit as st
import requests
import os
import numpy as np
import time

st.set_page_config(page_title="MLOps Optimizer", layout="wide")

st.title("🚀 MLOps Model Optimizer")
st.markdown("Upload your model and dataset, configure hyperparameter ranges, and watch the optimization in real-time.")

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
MLFLOW_URL = os.getenv("MLFLOW_URL", "http://localhost:5000")

# Sidebar for configuration
with st.sidebar:
    st.header("1. Upload Assets")
    uploaded_dataset = st.file_uploader("Upload Dataset (.npz)", type=["npz"])
    uploaded_model = st.file_uploader("Upload Model (.pkl, .pth, .h5)", type=["pkl", "pth", "h5"])
    
    if uploaded_dataset:
        if st.button("Upload Dataset"):
            files = {"file": uploaded_dataset.getvalue()}
            response = requests.post(f"{BACKEND_URL}/upload-data", files={"file": (uploaded_dataset.name, uploaded_dataset.getvalue())})
            if response.status_code == 200:
                st.session_state["data_path"] = response.json()["file_path"]
                st.success("Dataset uploaded!")
    
    if uploaded_model:
        if st.button("Upload Model"):
            response = requests.post(f"{BACKEND_URL}/upload-model", files={"file": (uploaded_model.name, uploaded_model.getvalue())})
            if response.status_code == 200:
                st.session_state["model_path"] = response.json()["model_path"]
                st.success("Model uploaded!")

# Main area
col1, col2 = st.columns(2)

with col1:
    st.header("2. Configure Optimization")
    model_name = st.selectbox("Model Type", ["SVC", "Random_Forest", "MLP", "XGBoost"])
    strategy = st.radio("Search Strategy", ["Grid", "Random", "Bayesian"])
    
    experiment_name = st.text_input("Experiment Name", value="Fashion_MNIST_Optim")
    
    st.subheader("Hyperparameter Space")
    if model_name == "SVC":
        c_range = st.multiselect("C values", [0.1, 1.0, 10.0, 100.0], default=[0.1, 1.0])
        kernels = st.multiselect("Kernels", ["linear", "rbf", "poly"], default=["linear", "rbf"])
        search_space = {"C": c_range, "kernel": kernels}
    elif model_name == "Random_Forest":
        n_est = st.slider("Min/Max estimators", 10, 500, (50, 200))
        depth = st.slider("Max depth", 1, 50, 10)
        search_space = {"n_estimators": list(range(n_est[0], n_est[1], 50)), "max_depth": [depth]}
    
    if st.button("Start Optimization", type="primary"):
        if "data_path" not in st.session_state:
            st.error("Please upload a dataset first!")
        else:
            config = {
                "model_name": model_name,
                "strategy": strategy.lower(),
                "search_space": search_space,
                "data_path": st.session_state["data_path"],
                "experiment_name": experiment_name
            }
            response = requests.post(f"{BACKEND_URL}/optimize", json=config)
            if response.status_code == 200:
                st.session_state["last_result"] = response.json()
                st.session_state["job_started"] = True
                st.rerun()

with col2:
    st.header("3. Results & Tracking")
    if "job_started" in st.session_state:
        st.write("🔄 Optimization in progress...")
        
        # In this specific setup, the backend call is synchronous (awaiting MCP)
        # So when the response comes, we have the result.
        if "last_result" in st.session_state:
            result = st.session_state["last_result"]
            if result.get("status") == "success":
                st.success("Optimization Complete!")
                data = result["result"]
                # MCP Tool returns data inside a 'content' list usually, 
                # but our wrapper in FastAPI simplified it.
                
                # If it's the raw MCP tool output
                if "content" in data:
                    res_json = json.loads(data["content"][0]["text"])
                else:
                    res_json = data
                
                st.metric("Best Score (Accuracy)", f"{res_json.get('best_score', 0):.4f}")
                st.subheader("Best Parameters")
                st.json(res_json.get("best_params", {}))
                
                if "run_id" in res_json:
                    st.info(f"MLflow Run ID: {res_json['run_id']}")
            else:
                st.error(f"Optimization failed: {result.get('message', 'Unknown error')}")
                
        st.markdown(f"[Go to MLflow UI]({MLFLOW_URL})")
    else:
        st.info("No optimization running. Configuration needed.")

st.divider()
st.caption("Agnostic MLOps Pipeline with MCP & MLflow")

st.divider()
st.header("4. Prediction")
st.markdown("Enter features as comma-separated values. One row = one prediction.")
feature_rows = st.text_area("Features", value="")
if st.button("Predict"):
    rows = [r.strip() for r in feature_rows.split("\n") if r.strip()]
    try:
        features = [
            [float(x) for x in row.split(",") if x.strip() != ""]
            for row in rows
        ]
        if not features:
            st.error("Please enter at least one row of numeric features.")
        else:
            response = requests.post(f"{BACKEND_URL}/predict", json={"features": features})
            if response.status_code == 200:
                st.json(response.json())
            else:
                st.error(f"Prediction failed: {response.text}")
    except ValueError:
        st.error("All feature values must be numeric.")
