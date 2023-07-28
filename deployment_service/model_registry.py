import mlflow
import mlflow.pyfunc
import argparse

def model_registry(run_id: str):
    mlflow.set_tracking_uri("http://localhost:5000") # read from config
    run_var = mlflow.get_run(run_id)
    # print(run_var)
    model_run_name = run_var.data.tags['mlflow.runName']

    with mlflow.start_run(run_name=model_run_name) as run:
        
        result = mlflow.register_model(
            f"runs:/{run_id}/model",
            model_run_name
        )

    #Transition to Staging stage

    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_run_name,
        version=1, #read from config
        stage="Staging"
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Model registry")
    parser.add_argument("--run_id", type=str, required=True)
    # parser.add_argument("--model_name", type=str, required=True)
    args = parser.parse_args()
    model_registry(args.run_id)