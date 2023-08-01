from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sys
sys.path.append(r'C:\Users\U1153226\Documents\pb_pivot2\mlflow\clinical_trial_repo\clinical_project-main\clinical_prj\project')
from deployment_service import monitoring_data
import argparse
import psycopg2
from pydantic import BaseModel
from uuid import uuid4
import torch
import mlflow
from preprocessing_service import data_preprocessing
from datetime import datetime
import pandas as pd

class Multilabel(BaseModel):
    assessment_text: str

def predict_service(run_id, data, stage):
    
    request_id = str(uuid4())

    # preprocess the feedback assessment_text/ production assessment text
    run = mlflow.get_run(run_id)
    model_name = run.data.params['_name_or_path']
    model_run_name = run.data.tags['mlflow.runName']
    
    model_uri = f"models:/{model_run_name}/{stage}"
    print('model_uri', model_uri)

    model = mlflow.pytorch.load_model(model_uri)
    print('ENTERED model', model)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if isinstance(data, pd.DataFrame):
        new_column_names = {'Assessment_Text': 'assessment_text'}
        data.rename(columns=new_column_names, inplace=True)
        input_texts = data["assessment_text"] if "assessment_text" in data else data["Assessment_Text"]
        input_texts = input_texts.tolist()
        
        predicted_labels_list = []
        def perform_inference(text):
            inputs = tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.sigmoid(logits)
                label_probs = [{'label': model.config.id2label[idx], 'score': prob.item()} for idx, prob in enumerate(probs[0])]
            return label_probs, probs
        
        preds ={}
        preds_list =[]
        for text in input_texts:
            label_probs, probs = perform_inference(text)
            predicted_labels = (probs > 0.5).long()
            pred_prob_for_each_text = predicted_labels[0].tolist()
            predicted_labels_list.append(pred_prob_for_each_text)
            preds["text"] = label_probs
            preds_list.append(preds)

        
        labels = [label for label in data.columns.to_list() if label not in ["assessment_text"]]
        actual_labels_list = data[labels].values.tolist()
        print('actual_labels_list', actual_labels_list)
    
    else:
        predicted_labels_list = []
        actual_labels_list = []
        inputs = tokenizer(data, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.sigmoid(logits)
            label_probs = [{'label': model.config.id2label[idx], 'score': prob.item()} for idx, prob in enumerate(probs[0])]
        
        input_texts = data
        preds_list = [label_probs]

    print('Predicted Probs------------------------',label_probs)

    # monitoring_data.log_data(
    #     request_id, 
    #     input_texts, 
    #     preds_list, 
    #     "logs", 
    #     model_info, 
    #     )
    return predicted_labels_list, actual_labels_list, label_probs

