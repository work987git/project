import os
import pandas as pd
import datetime
import subprocess
import yaml

class DataProcessor:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)

    def load_config(self, config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config

    def preprocess_data(self):
        path = self.config['old_training_data_path']
        path = os.path.join(path, "training.csv")
        train_data_old = pd.read_csv(path)
        train_data_old.columns = train_data_old.columns.str.lower()
        time_stamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

        #TODO DVC
        training_history_path = self.config['training_data_history']
        if not os.path.exists(training_history_path):
            os.makedirs(training_history_path)
        old_train_csv = "training_" + time_stamp + ".csv"
        training_history_path = os.path.join(training_history_path, old_train_csv)
        train_data_old.to_csv(training_history_path, index=False)

        feedback_data = pd.read_csv(csv_file_path)
        new_train = pd.concat([train_data_old, feedback_data], ignore_index=True)
        new_train.to_csv(path, index=False)

    def run_retraining(self):
        # Call train.py for retraining
        training_script_path = r'C:\Users\U1153226\Documents\pb_pivot2\mlflow\clinical_trial_repo\clinical_project-main\clinical_project-main\app_train\train.py'
        config_training_path = r'C:\Users\U1153226\Documents\pb_pivot2\mlflow\clinical_trial_repo\clinical_project-main\clinical_project-main\app_train\config.yaml'
        subprocess.run(["python", training_script_path, "--config_path", config_training_path], check=True)

# Example usage:
config_path = "path/to/config.yaml"
csv_file_path = "path/to/feedback_data.csv"

# Create an instance of the DataProcessor class
data_processor = DataProcessor(config_path)

# Preprocess the data
data_processor.preprocess_data()

# Run the retraining
data_processor.run_retraining()
