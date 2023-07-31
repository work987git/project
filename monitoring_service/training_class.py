import os
import pandas as pd
import datetime
import subprocess
import yaml

class DataProcessor:
    def __init__(self, config_path, feedback_csv):
        self.feedback_csv = feedback_csv
        self.config_path = config_path
        self.config = self.load_config(config_path)
        print(self.config)
        self.training_script_path = r'C:\Users\U1153226\Documents\pb_pivot2\mlflow\clinical_trial_repo\clinical_project-main\clinical_project-main\app_train\train.py'
        self.config_training_path = r'C:\Users\U1153226\Documents\pb_pivot2\mlflow\clinical_trial_repo\clinical_project-main\clinical_project-main\app_train\config.yaml'


    def load_config(self, config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config


    def versioning_data(self, csv_file_path):
        # try:
        tag = self.config['dvc_latest_version'] 
        git_checkout_command = ['git', 'checkout', tag]
        subprocess.run(git_checkout_command, capture_output=True, text=True, check=True)

        # Pull the latest csv 
        dvc_pull_command = ['dvc', 'pull', '--force']
        subprocess.run(dvc_pull_command, capture_output=True, text=True, check=True)

        path = self.config['training_data_path']
        path = os.path.join(path, "training.csv")
        train_data_old = pd.read_csv(path)
        train_data_old.columns = train_data_old.columns.str.lower()

        feedback_data = pd.read_csv(csv_file_path)
        new_train = pd.concat([train_data_old, feedback_data], ignore_index=True)

        # Add the updated file to DVC tracking
        dvc_add_command = ['dvc', 'add', path]
        subprocess.run(dvc_add_command, capture_output=True, text=True, check=True)

        # Push the changes to the remote DVC storage
        dvc_push_command = ['dvc', 'push']
        subprocess.run(dvc_push_command, capture_output=True, text=True, check=True)

        # Stage the changes and commit them using Git
        git_add_command = ['git', 'add', self.config['training_dvc']]
        subprocess.run(git_add_command, capture_output=True, text=True, check=True)

        time_stamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        
        git_commit_command = ['git', 'commit', '-m', f"training_csv_dvc_{time_stamp}"]
        subprocess.run(git_commit_command, capture_output=True, text=True, check=True)

        # Push the changes to the remote Git repository
        git_push_command = ['git', 'push']
        subprocess.run(git_push_command, capture_output=True, text=True, check=True)

        # Tag the commit with the DVC version tag
        
        dvc_new_version_tag = "training_" + time_stamp
        git_tag_command = ['git', 'tag', dvc_new_version_tag]
        subprocess.run(git_tag_command, capture_output=True, text=True, check=True)

        # Push the tag to the remote Git repository
        git_push_tag_command = ['git', 'push', '--tags']
        subprocess.run(git_push_tag_command, capture_output=True, text=True, check=True)
        
        self.config['dvc_latest_version'] = dvc_new_version_tag
        with open(self.config_path, "w") as f:
            yaml.dump(self.config, f)
                    
        # except subprocess.CalledProcessError as e:
        #     print(f"Error occurred: {e.stderr}")
        # except Exception as e:
        #     print(f"An error occurred: {str(e)}")


    def run_retraining(self):
        # Call train.py for retraining
        self.versioning_data(self.feedback_csv)
        subprocess.run(["python", self.training_script_path, "--config_path", self.config_training_path], check=True)

feed_csv=r'C:\Users\U1153226\Documents\pb_pivot2\mlflow\clinical_trial_repo\clinical_project-main\clinical_prj\project\feedback_csv\feedback_914369bc62b3495da32a13705bdab7b0_20230731183528.csv'
config_path = r'C:\Users\U1153226\Documents\pb_pivot2\mlflow\clinical_trial_repo\clinical_project-main\clinical_prj\project\monitoring_service\config.yaml'
data_processor = DataProcessor(config_path, feed_csv )
data_processor.run_retraining()

        # git checkout tag_name
        # dvc pull
        
        # dvc add path
        # dvc push
        # git add path
        # git commit -m "data"+time_stamp
        # git push
        # git tag dvc_new_version_tag
        # git push --tag