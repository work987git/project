import mlflow
import data_utils
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import yaml
from typing import Tuple, Dict
import sys
sys.path.append(r'C:\Users\U1153226\Documents\pb_pivot2\mlflow\clinical_trial_repo\clinical_project-main\clinical_project-main')
from prediction_service import prediction_multi_label
from preprocessing_service import calculate_metrics
import pandas as pd

def load_config(config_path: str) -> Tuple[int, Dict[str, str]]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def calculate_metrics_test(run_id, ground_truth, prediction):
    labels = ground_truth
    predicted_labels = prediction
    accuracy = accuracy_score(labels, predicted_labels)
    precision = precision_score(labels, predicted_labels, average='micro')
    recall = recall_score(labels, predicted_labels, average='micro')
    f1 = f1_score(labels, predicted_labels, average='micro')



def fetch(config_path: str, run_id: str, port: int):
    config = load_config(config_path)
    mlflow.set_tracking_uri("http://localhost:5000") # read from config
    run = mlflow.get_run(run_id)
    model_run_name = run.data.tags['mlflow.runName']
    stage = 'Production' #'Staging'  read from config
    experiment_id = run.info.experiment_id
    experiment_name = mlflow.get_experiment(experiment_id).name
    # load test dataset
    # dataset = data_utils.load_dataset(config['processed_data_path'])
    # X_test = dataset['test']

    X_test = pd.read_csv(r'C:\Users\U1153226\Documents\pb_pivot2\mlflow\clinical_trial_repo\clinical_project-main\clinical_project-main\data\testing.csv')
    model_info = {
        "model_name": model_run_name,
        "experiment_name": experiment_name,
        "run_id": run_id,
        "port": port
    }

    # Make predictions on test dataset that would be stored using mlflow
    # call prediction service - covers fetching the model, calling preprocessing service and making a prediction
    # call calculate_metrics() and pass the returned value from prediction service
    predicted_labels_list, actual_labels_list, label_probs = prediction_multi_label.predict_service(run_id, X_test, model_info, stage)
    calculate_metrics.calculate_metrics_multi_label(run_id, actual_labels_list, predicted_labels_list, flag = 'test')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the configuration file",
    )
    parser.add_argument("--run_id", type = str, required = True)
    parser.add_argument("--port", type=int, default=8080, help="Port number")
    args = parser.parse_args()
    fetch(args.config_path, args.run_id, args.port)

