import yaml
import threading
import argparse
import subprocess
class ModelDeployment:
    def __init__(
            self,
            run_id, 
            host, 
            port, 
            mlflow_tracking_uri,
            logging_db):
        self.run_id = run_id
        self.host = host
        self.port = port
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.logging_db = logging_db

    def run(self):
        cmd = f"python deployment_service/app.py --run_id {self.run_id} --host {self.host} --port {self.port} --mlflow_tracking_uri {self.mlflow_tracking_uri} --logging_db {self.logging_db}"
        subprocess.Popen(cmd, shell=True)
class DeploymentManager:
    def __init__(self, config_file):
        self.deployments = []

        if isinstance(config_file, dict):
            configs = [config_file]
        else:
            with open(config_file, "r") as file:
                configs = yaml.safe_load(file)

        for config in configs:
            deployment = ModelDeployment(
                run_id=config["run_id"],
                host=config["host"],
                port=config["port"],
                mlflow_tracking_uri=config["mlflow_tracking_uri"],
                logging_db=config["logging_db"]
            )
            self.deployments.append(deployment)

    def deploy_all(self):
        threads = []
        for deployment in self.deployments:
            thread = threading.Thread(target=deployment.run)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--config_file", 
        type=str, 
        default=r"C:\Users\U1153226\Documents\pb_pivot2\mlflow\clinical_trial_repo\clinical_project-main\clinical_prj\project\deployment_service\config.yaml", 
        help="Path to the YAML configuration file")
    
    args = parser.parse_args()

    manager = DeploymentManager(args.config_file)
    manager.deploy_all()
