import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
import ast
import mlflow
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime
import sys
sys.path.append(r'C:\Users\U1153226\Documents\pb_pivot2\mlflow\clinical_trial_repo\clinical_project-main\clinical_project-main')

from deployment_service import table_creation_logging
from preprocessing_service import calculate_metrics
from feedback_service import feedback
from datasets import Dataset
import os
import datetime
import subprocess
import yaml
from typing import Dict, Tuple
from app_train import data_utils
import pandas as pd

def load_config(config_path: str) -> Tuple[int, Dict[str, str]]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def model_monitor(run_id, config_path):
    mlflow.set_tracking_uri("http://localhost:5000") #TODO read from config
    run = mlflow.get_run(run_id)

    # call feedback service
    csv_file_path = feedback.feedback_service(run_id)
    print('PREDICTIONS ON FEEDBACK DONE')

    # compare feedback and testing metrics to detect model drift
    if (run.data.metrics['test_f1'] > run.data.metrics['feedback_f1'] or run.data.metrics['test_accuracy'] > run.data.metrics['feedback_accuracy'] 
    or run.data.metrics['test_precision'] > run.data.metrics['feedback_precision']):
        print('MODEL DRIFT DETECTED')
        # append the feedback table to training data
        #TODO try and except out of scope interaction
        config = load_config(config_path)
        path = config['old_training_data_path'] #trycatch
        path = os.path.join(path, "training.csv")
        train_data_old = pd.read_csv(path)
        train_data_old.columns = train_data_old.columns.str.lower()
        time_stamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        #TODO DVC
        training_history_path = config['training_data_history']
        if not os.path.exists(training_history_path):
            os.makedirs(training_history_path)
        old_train_csv = "training_"+time_stamp+".csv"
        training_history_path = os.path.join(training_history_path,old_train_csv)
        train_data_old.to_csv(training_history_path,index = False)
        feedback_data = pd.read_csv(csv_file_path)
        new_train = pd.concat([train_data_old, feedback_data], ignore_index=True)
        new_train.to_csv(path,index = False)
        # call train.py for retraining
        training_script_path = r'C:\Users\U1153226\Documents\pb_pivot2\mlflow\clinical_trial_repo\clinical_project-main\clinical_project-main\app_train\train.py'
        config_training_path = r'C:\Users\U1153226\Documents\pb_pivot2\mlflow\clinical_trial_repo\clinical_project-main\clinical_project-main\app_train\config.yaml'
        subprocess.run(["python", training_script_path, "--config_path", config_training_path], check=True)
    else:
        print('NO MODEL DRIFT DETECTED')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=str, required=True)
    parser.add_argument('--config_path', type=str, required=True)
    args = parser.parse_args()

    model_monitor(args.run_id, args.config_path)
