import mlflow
import os

def get_run_info(run_id, mlflow_tracking_uri):

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    return dict(mlflow.get_run(run_id=run_id).info)

def download_artifacts(run_id, mlflow_tracking_uri):
    
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)

    artifact_uri = run.info.artifact_uri
    output_path = f"models/{run_id}"

    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
        print(f"Model {run_id} doesn't exist locally. Downloading...")
        mlflow.tracking.artifact_utils._download_artifact_from_uri(
            artifact_uri, 
            output_path
            )
    else:    
        print(f"Model {run_id} exists locally.")

    return output_path + "/artifacts/model"