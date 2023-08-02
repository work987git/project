import streamlit as st
import mlflow
import sys
sys.path.append(r'C:\Users\U1153226\Documents\pb_pivot2\mlflow\clinical_trial_repo\clinical_project-main\clinical_prj\project')

from deployement_manager import DeploymentManager

 

mlflow_tracking_uri=st.text_input("MLFLOW_TRACKING_URI:", value="postgresql+psycopg2://postgres:postgres@127.0.0.1:5432/postgres")

run_id = st.text_input("Run id:", value= 'fbc7ca62e8ed4bf480df35cc635ed9bb')

 

if run_id and mlflow_tracking_uri:
    # model_name: able-smelt-83
    # experiment_name: Experiment_on_Bio_ClinicalBERT
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    run = mlflow.get_run(run_id)
    
    model_name = run.data.tags['mlflow.runName']
    experiment_id = run.info.experiment_id
    experiment_name = mlflow.get_experiment(experiment_id).name
    port= st.number_input("Port", value=8080)
    host= st.text_input("Host id:", value= '127.0.0.1')
    logging_db=st.text_input("Logging DB URI: ","postgresql+psycopg2://postgres:postgres@127.0.0.1:5432/postgres")
    deployment_config = {

        "run_id": run_id,

        "logging_db": logging_db,

        "port": port,

        "host": host,
        "model_name" : model_name,
    "experiment_name"  : experiment_name,
    "mlflow_tracking_uri": mlflow_tracking_uri
    }

 

    if st.button("Deploy"):
        st.write("Deploying model with following config:")
        st.json(deployment_config)
        DeploymentManager(deployment_config).deploy_all()
        st.success("Model is deployed successfully")