import mlflow
import mlflow.pyfunc
import argparse

def model_production(run_id):
    mlflow.set_tracking_uri("http://localhost:5000") # read from config
    run = mlflow.get_run(run_id)

    run_name = run.data.params['_name_or_path']
    model_run_name = run.data.tags['mlflow.runName']

    #Transition to Production stage

    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_run_name,
        version=1, #read from config
        stage="Production" #read from config
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Model registry")
    parser.add_argument("--run_id", type=str, required=True)
    # parser.add_argument("--model_name", type=str, required=True)
    args = parser.parse_args()
    model_production(args.run_id)