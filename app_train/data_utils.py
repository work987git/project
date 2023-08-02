import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
import yaml
from pathlib import Path
import os
from pandas_profiling import ProfileReport
from typing import Tuple, Dict
import mlflow

def load_dataset(path):
    project_root = Path(__file__).resolve().parents[2]
    training_path = project_root  / path / "training.csv"
    testing_path = project_root  / path / "testing.csv"
    validation_path = project_root  / path / "validation.csv"
    train_df = pd.read_csv(training_path)
    test_df = pd.read_csv(testing_path)
    validation_df = pd.read_csv(testing_path)
    train_df.columns = train_df.columns.str.lower()
    test_df.columns = test_df.columns.str.lower()
    validation_df.columns = validation_df.columns.str.lower()
    train = Dataset.from_pandas(train_df)
    test = Dataset.from_pandas(test_df)
    validation = Dataset.from_pandas(validation_df)
    return DatasetDict({'train': train, 'test': test, 'validation': validation})

def preprocess_data(examples, tokenizer, labels, config):
    text = examples["assessment_text"]
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=config['max_length'])
    labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
    labels_matrix = np.zeros((len(text), len(labels)))

    for idx, label in enumerate(labels):
        labels_matrix[:, idx] = labels_batch[label]

    encoding["labels"] = labels_matrix.tolist()

    return encoding

def load_config(config_path: str) -> Tuple[int, Dict[str, str]]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def create_and_log_profiling(path, run_id):
    dataset_dict = load_dataset(path)
    train_dataset = dataset_dict['train']
    train_df = train_dataset.to_pandas()
    config = r'C:\Users\U1153226\Documents\pb_pivot2\mlflow\clinical_trial_repo\clinical_project-main\clinical_prj\project\monitoring_service\config.yaml'
    profiling_html_dir = r'C:\Users\U1153226\Documents\pb_pivot2\mlflow\clinical_trial_repo\clinical_project-main\clinical_prj\project\profiling'
    # Create the output directory if it doesn't exist
    if not os.path.exists(profiling_html_dir):
        os.makedirs(profiling_html_dir)

    #TODO add the version name of the data to the profiling report
    profile = ProfileReport(train_df, title='Pandas Profiling Report', explorative=True)

    config_dvc_version = load_config(config)
    config_dvc_version= config_dvc_version['dvc_latest_version']
    html_name = config_dvc_version+".html"
    # Save the report to the output directory
    profile_html_path = os.path.join(profiling_html_dir, html_name)
    profile.to_file(profile_html_path)
    if run_id:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifact(profile_html_path, artifact_path="profiling_reports")
    return True