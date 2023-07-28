import mlflow
import sys
sys.path.append(r'C:\Users\U1153226\Documents\pb_pivot2\mlflow\clinical_trial_repo\clinical_project-main\clinical_project-main')
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

def convert_feeds(run_id, feedtable):
    feedback_csv_dir = 'feedback_csv' # config
    feedtable['feedback'] = feedtable['feedback'].apply(ast.literal_eval)
    for key in feedtable['feedback'][0].keys():
        feedtable[key] = feedtable['feedback'].apply(lambda x: x.get(key, 0))
    cols_to_drop = ['id', 'model_id', 'feedback']
    feedtable.drop(columns=cols_to_drop, inplace=True)
    date = dt.datetime.now().strftime('%Y%m%d%H%M%S')
    os.makedirs(feedback_csv_dir, exist_ok=True)
    csv_file_path = Path(feedback_csv_dir) / f"feedback_{run_id}_{date}.csv"
    feedtable.to_csv(csv_file_path,index = False)
    # with open(csv_file_path, 'w') as f:
    #     f.write(feedtable)
    print(feedtable)
    return feedtable, csv_file_path

def feedback_service(run_id):
    mlflow.set_tracking_uri("http://localhost:5000") # config

    #  retrive feedback from db
    feed_table = table_creation_logging.retrive_feedback(run_id) # wait for api file to get developed

    # convert feedback into dataframe and csv form - diff func(to confirm if csv is aligned with schema)
    feed_table, csv_file_path = convert_feeds(run_id, feed_table)

    # call prediction service to predict on feedback dataset
    predicted_labels_list, actual_labels_list, label_probs = prediction_multi_label.predict_service(run_id, feed_table, None, stage='Production')
    #logs the metrics in mlflow
    calculate_metrics.calculate_metrics_multi_label(run_id, actual_labels_list, predicted_labels_list, flag = 'feedback')

    return csv_file_path

# based on the model id all prediction service - in prediction service we will save the logs in mlflow as well
# we will calculate the metrics on dataset and store it using mlflow under the name "feedback_runid_model_name_date"

# once done return True to monitoring service so that it fetches the metrics for both feedback and 
# evaluation dataset and compares to detect model drift

# usecases that our architecture would solve
# components
# communication among components


