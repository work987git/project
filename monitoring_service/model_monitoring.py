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
sys.path.append(r'C:\Users\U1153226\Documents\pb_pivot2\mlflow\clinical_trial_repo\clinical_project-main\clinical_prj\project')

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
from training_class import DataProcessor

def model_monitor(run_id, config_path):
    training_cls = DataProcessor(config_path)
    mlflow.set_tracking_uri("http://localhost:5000") #TODO read from config
    run = mlflow.get_run(run_id)

    # call feedback service
    feedback_csv_file_path = feedback.feedback_service(run_id)
    print('PREDICTIONS ON FEEDBACK DONE')

    # compare feedback and testing metrics to detect model drift
    if (run.data.metrics['test_f1'] > run.data.metrics['feedback_f1'] 
        or run.data.metrics['test_accuracy'] > run.data.metrics['feedback_accuracy'] 
        or run.data.metrics['test_precision'] > run.data.metrics['feedback_precision']):
        print('MODEL DRIFT DETECTED')
        training_cls.run_retraining(feedback_csv_file_path, config_path)

    else:
        training_cls.run_retraining(feedback_csv_file_path, config_path)
        print('NO MODEL DRIFT DETECTED')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=str, required=True)
    parser.add_argument('--config_path', type=str, required=True)
    args = parser.parse_args()

    model_monitor(args.run_id, args.config_path)
