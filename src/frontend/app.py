import streamlit as st
import requests
import os
import json
import numpy as np
import time

st.set_page_config(page_title="MLOps Optimizer", layout="wide")

st.title("🚀 MLOps Model Optimizer")
st.markdown("Upload your model and dataset, configure hyperparameter ranges, and watch the optimization in real-time.")

BACKEND_URL = os.getenv("BACKEND_URL", "https://mlops-backend-api-2ahwjpnqfq-ew.a.run.app")
MLFLOW_URL = os.getenv("MLFLOW_URL", "http://mlflow-server-mlops-sadiya-mourad.francecentral.azurecontainer.io:5000")

# Sidebar for configuration
with st.sidebar:
    st.header("1. Upload Dataset")
    uploaded_dataset = st.file_uploader("Upload Dataset (.npz)", type=["npz"])
    
    if uploaded_dataset:
        if st.button("Upload Dataset"):
            files = {"file": uploaded_dataset.getvalue()}
            response = requests.post(f"{BACKEND_URL}/upload-data", files={"file": (uploaded_dataset.name, uploaded_dataset.getvalue())})
            if response.status_code == 200:
                st.session_state["data_path"] = response.json()["file_path"]
                st.success("Dataset uploaded!")
    
    st.divider()
    st.header("2. Select Model")
    
    # Fetch models from MLflow
    try:
        models_response = requests.get(f"{BACKEND_URL}/models")
        if models_response.status_code == 200:
            models_data = models_response.json()
            if models_data["status"] == "success" and models_data["models"]:
                model_options = []
                model_uris = {}
                
                for model in models_data["models"]:
                    for version in model["versions"]:
                        label = f"{model['name']} v{version['version']} ({version['stage']})"
                        uri = version.get("stage_uri") or version["uri"]
                        model_options.append(label)
                        model_uris[label] = uri
                
                selected_model = st.selectbox("Available Models", model_options)
                st.session_state["selected_model_uri"] = model_uris.get(selected_model)
                st.info(f"URI: `{st.session_state['selected_model_uri']}`")
            else:
                st.warning("No models found in MLflow. Train a model first!")
        else:
            st.error("Failed to fetch models from MLflow")
    except Exception as e:
        st.error(f"Error connecting to backend: {e}")

# Main area
col1, col2 = st.columns(2)

with col1:
    st.header("3. Configure Optimization")
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
    st.header("4. Results & Tracking")
    if "job_started" in st.session_state:
        st.write("🔄 Optimization in progress...")
        
        # In this specific setup, the backend call is synchronous (awaiting MCP)
        # So when the response comes, we have the result.
        if "last_result" in st.session_state:
            result = st.session_state["last_result"]
            if result.get("status") == "success":
                st.success("Optimization Complete!")
                res_data = result["result"]
                
                st.metric("Best Score (Accuracy)", f"{res_data.get('best_score', 0):.4f}")
                st.subheader("Best Parameters")
                st.json(res_data.get("best_params", {}))
                
                if "run_id" in res_data:
                    st.info(f"MLflow Run ID: {res_data['run_id']}")
            else:
                st.error(f"Optimization failed: {result.get('message', 'Unknown error')}")
                
        # Link to MLflow experiment
        if "last_result" in st.session_state and st.session_state["last_result"].get("status") == "success":
            experiment_id = st.session_state["last_result"]["result"].get("experiment_id")
            if experiment_id:
                st.markdown(f"[📊 View Experiment in MLflow]({MLFLOW_URL}/#/experiments/{experiment_id})")
            else:
                st.markdown(f"[Go to MLflow UI]({MLFLOW_URL})")
        else:
            st.markdown(f"[Go to MLflow UI]({MLFLOW_URL})")
    else:
        st.info("No optimization running. Configuration needed.")

st.divider()
st.caption("Agnostic MLOps Pipeline with MCP & MLflow")

st.divider()
st.header("5. Prediction")
st.markdown("Enter features as comma-separated values. One row = one prediction.")

if "selected_model_uri" in st.session_state:
    st.info(f"Using model: `{st.session_state['selected_model_uri']}`")

feature_rows = st.text_area("Features", value="")
if st.button("Predict"):
    if "selected_model_uri" not in st.session_state:
        st.error("Please select a model from the sidebar first!")
    else:
        rows = [r.strip() for r in feature_rows.split("\n") if r.strip()]
        try:
            features = [
                [float(x) for x in row.split(",") if x.strip() != ""]
                for row in rows
            ]
            if not features:
                st.error("Please enter at least one row of numeric features.")
            else:
                payload = {
                    "features": features,
                    "model_uri": st.session_state["selected_model_uri"]
                }
                response = requests.post(f"{BACKEND_URL}/predict", json=payload)
                if response.status_code == 200:
                    result = response.json()
                    st.success("Prediction successful!")
                    st.json(result)
                else:
                    st.error(f"Prediction failed: {response.text}")
        except ValueError:
            st.error("All feature values must be numeric.")
