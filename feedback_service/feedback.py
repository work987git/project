import mlflow
import sys
sys.path.append(r'C:\Users\U1153226\Documents\pb_pivot2\mlflow\clinical_trial_repo\clinical_project-main\clinical_prj\project')
from deployment_service import table_creation_logging
import ast
import os
from pathlib import Path
import datetime as dt
from datetime import datetime
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from prediction_service import prediction_multi_label
from preprocessing_service import calculate_metrics

import pandas as pd

# class TrainingDataset:
#     def __init__(self, df):
#         self.df = df
#         for column in self.df.columns:
#             setattr(self, column, self.df[column])
            
class TrainingDataset():
    assessment_text: str
    endoscopy: int = 0 
    colonoscopy: int = 0 
    lumbar_puncture: int = 0 
    mri: int = 0
    pet: int = 0
    xray: int = 0
    
    

# def convert_feeds(run_id, feedtable):
#     feedback_csv_dir = 'feedback_csv' # config
#     feedtable['feedback'] = feedtable['feedback'].apply(ast.literal_eval)
#     for key in feedtable['feedback'][0].keys():
#         feedtable[key] = feedtable['feedback'].apply(lambda x: x.get(key, 0))
#     cols_to_drop = ['id', 'model_id', 'feedback']
#     feedtable.drop(columns=cols_to_drop, inplace=True)
#     date = dt.datetime.now().strftime('%Y%m%d%H%M%S')
#     os.makedirs(feedback_csv_dir, exist_ok=True)
#     csv_file_path = Path(feedback_csv_dir) / f"feedback_{run_id}_{date}.csv"
#     feedtable.to_csv(csv_file_path,index = False)
#     # with open(csv_file_path, 'w') as f:
#     #     f.write(feedtable)
#     print(feedtable)
#     return feedtable, csv_file_path


def convert_feeds(run_id, feedtable):
    
    feedback_csv_dir = 'feedback_csv' # config
    
    data_rows = []
    i = 0
    for label in feedtable['feedback_labels']:
        training_instance = TrainingDataset()
        training_instance.assessment_text=feedtable['query_text'][i]
    # Convert the label to lowercase to make it case-insensitive
        label = label.lower()
        
        
        # Set the properties based on the presence of specific procedures
        training_instance.endoscopy = 1 if 'endoscopy' in label else training_instance.endoscopy
        training_instance.colonoscopy = 1 if 'colonoscopy' in label else training_instance.colonoscopy
        training_instance.lumbar_puncture = 1 if 'lumbar_puncture' in label else training_instance.lumbar_puncture
        training_instance.mri = 1 if 'mri' in label else training_instance.mri
        training_instance.pet = 1 if 'pet' in label else training_instance.pet
        training_instance.xray = 1 if 'xray' in label else training_instance.xray
        
        print(training_instance.assessment_text, 
              training_instance.endoscopy ,
        training_instance.colonoscopy ,
        training_instance.lumbar_puncture,
        training_instance.mri ,
        training_instance.pet ,
        training_instance.xray)
        
        data_rows.append({
        'assessment_text': training_instance.assessment_text,
        'endoscopy': training_instance.endoscopy,
        'colonoscopy': training_instance.colonoscopy,
        'lumbar_puncture': training_instance.lumbar_puncture,
        'mri': training_instance.mri,
        'pet': training_instance.pet,
        'xray': training_instance.xray
        })
        i+=1
        
    feedtable = pd.DataFrame(data_rows)
    date = dt.datetime.now().strftime('%Y%m%d%H%M%S')
    os.makedirs(feedback_csv_dir, exist_ok=True)
    csv_file_path = Path(feedback_csv_dir) / f"feedback_{run_id}_{date}.csv"
    feedtable.to_csv(csv_file_path,index = False)

    return feedtable, csv_file_path

def feedback_service(run_id):
    mlflow.set_tracking_uri("http://localhost:5000") # config

    #  retrive feedback from db
    feed_table = table_creation_logging.retrive_feedback(run_id) # wait for api file to get developed

    # convert feedback into dataframe and csv form - diff func(to confirm if csv is aligned with schema)
    feed_table, csv_file_path = convert_feeds(run_id, feed_table)

    # call prediction service to predict on feedback dataset
    predicted_labels_list, actual_labels_list, label_probs = prediction_multi_label.predict_service(run_id, feed_table, stage='Production')
    # #logs the metrics in mlflow
    calculate_metrics.calculate_metrics_multi_label(run_id, actual_labels_list, predicted_labels_list, flag = 'feedback')

    return csv_file_path


# feedback_service('fbc7ca62e8ed4bf480df35cc635ed9bb')