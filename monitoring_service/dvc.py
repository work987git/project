# import os
# import subprocess
# import yaml
# from typing import Dict, Tuple
# from feedback_service import feedback
# import mlflow


def initialize_dvc_if_not_initialized():
    if not os.path.exists('.dvc'):
        subprocess.run(['dvc', 'init'])
        print("DVC has been initialized.")
    else:
        print("DVC is already initialized.")
        
# def dvc_add_file_or_directory(path):
#     try:
#         #TODO read from config
#         path = r'C:\Users\U1153226\Documents\pb_pivot2\mlflow\clinical_trial_repo\clinical_project-main\clinical_prj\project\data'
#         # Construct the DVC add command
#         dvc_add_command = ['dvc', 'add', path]
        
#         # Run the DVC add command as a subprocess
#         result = subprocess.run(dvc_add_command, capture_output=True, text=True, check=True)

#         # Print the DVC output (optional)
#         print(result.stdout)
        
#         print(f"Successfully added {path} to DVC tracking.")
#     except subprocess.CalledProcessError as e:
#         # If the DVC command fails, print the error message
#         print(f"Error occurred: {e.stderr}")
#     except Exception as e:
#         # Handle any other exceptions that might occur
#         print(f"An error occurred: {str(e)}")


# def generate_training_datasets():
#     # Step 2: Your dataset generation logic here
#     # Save the datasets with unique names in the 'data' directory
#     print()

# def add_datasets_to_dvc(data_dir):
#     # Step 3: Get a list of files in the data directory
#     dataset_files = os.listdir(data_dir)

#     for dataset_file in dataset_files:
#         # Step 4: Use dvc add to add each dataset file to DVC
#         dataset_path = os.path.join(data_dir, dataset_file)
#         subprocess.run(["dvc", "add", dataset_path])

# if __name__ == "__main__":
#     data_directory = "data"  # Replace with the path to your data directory

#     # Step 1: Initialize DVC in the project directory
#     initialize_dvc()

#     # Step 2: Generate Training Datasets
#     generate_training_datasets()

#     # Step 3: Automate DVC Dataset Addition
#     add_datasets_to_dvc(data_directory)

#     # Step 4: Commit the DVC changes
#     subprocess.run(["git", "add", "data/*.dvc"])
#     subprocess.run(["git", "commit", "-m", "Added new training datasets to DVC"])





import os
import subprocess
import pandas as pd
import datetime

def preprocess_data(csv_file_path):
    try:
        # TODO: Preprocessing
        path = r"C:\Users\U1153226\Documents\pb_pivot2\mlflow\clinical_trial_repo\clinical_project-main\clinical_prj\project\data\training.csv"
        train_data_old = pd.read_csv(path)
        train_data_old.columns = train_data_old.columns.str.lower()

        feedback_data = pd.read_csv(csv_file_path)
        new_train = pd.concat([train_data_old, feedback_data], ignore_index=True)
        new_train.to_csv(path, index=False)

        # Get the current timestamp
        time_stamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

        # TODO: DVC
        # Add the updated file to DVC tracking
        dvc_add_command = ['dvc', 'add', path]
        subprocess.run(dvc_add_command, capture_output=True, text=True, check=True)

        # Push the changes to the remote DVC storage
        dvc_push_command = ['dvc', 'push']
        subprocess.run(dvc_push_command, capture_output=True, text=True, check=True)

        # Tag the data version with DVC
        dvc_tag_command = ['dvc', 'tag', f"training_{time_stamp}"]
        subprocess.run(dvc_tag_command, capture_output=True, text=True, check=True)

    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e.stderr}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    csv_file_path = r"C:\Users\U1153226\Documents\pb_pivot2\mlflow\clinical_trial_repo\clinical_project-main\clinical_prj\project\data\validation.csv"  # Replace with the actual path to the CSV file
    preprocess_data(csv_file_path)
