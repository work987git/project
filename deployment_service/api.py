from fastapi import FastAPI, File
from pydantic import BaseModel, Field
import pickle
import numpy as np
import pandas as pd
from io import StringIO
import requests
import psycopg2
import datetime
import json
from uuid import uuid4
import sys
from typing import List, Tuple
import mlflow.pytorch
import mlflow
from monitoring_data import log_data
import argparse
import os
import uvicorn
import torch
import requests
from pydantic import BaseModel
from uuid import uuid4
import table_creation_logging
from lime.lime_text import LimeTextExplainer

print(sys.path)
sys.path.append(r'C:\Users\U1153226\Documents\pb_pivot2\mlflow\clinical_trial_repo\clinical_project-main\clinical_project-main')
from prediction_service import prediction_multi_label

from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI()

def load_model_and_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return model, tokenizer


@app.get('/')
async def root():
    return  {"message":"docs#"}

# Establish a connection with the PostgreSQL database
conn = psycopg2.connect(
    host="127.0.0.1",
    database="postgres",
    user="postgres",
    password="postgres"
)

cur = conn.cursor()

class Multilabel(BaseModel):
    assessment_text: str

class ExplanationResult(BaseModel):
    explanation: List[Tuple[str, float]]

class FeedbackRequest(BaseModel):
    assessment_text: str
    endoscopy: int = 0
    colonoscopy: int = 0
    lumbar_puncture: int = 0
    mri: int = 0
    pet: int = 0
    xray: int = 0

# def fetch_runs(run_id):
#     run = mlflow.get_run(args.run_id)
#     model_run_name = run.data.tags['mlflow.runName']
#     experiment_id = run.info.experiment_id
#     experiment_name = mlflow.get_experiment(experiment_id).name

#     return model_run_name, experiment_name

@app.on_event("startup")
async def startup_event():
    table_creation_logging.create_database(conn)
    model_info = {
        "model_name": args.model_name,
        "experiment_name": args.experiment_name,
        "run_id": args.run_id,
        "host": args.host,
        "port": args.port
    }
    table_creation_logging.log_events(conn, "startup", model_info, "active")

@app.on_event("shutdown")
def shutdown_event():
    model_info = {
        "model_name": args.model_name,
        "experiment_name": args.experiment_name,
        "run_id": args.run_id,
        "host": args.host,
        "port": args.port
    }
    table_creation_logging.log_events("shutdown", model_info, "inactive")

def generate_lime_explanation(model, tokenizer, text, num_labels, num_features=10, num_samples=500):
    def predictor(texts):
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits)
        return probs.numpy()

    explainer = LimeTextExplainer(class_names=[model.config.id2label[i] for i in range(num_labels)])
    exp = explainer.explain_instance(text, predictor, num_features=num_features, top_labels=num_labels, num_samples=num_samples)
    return exp.as_list()

@app.post("/feedback")
async def collect_feedback(feedback: FeedbackRequest):
    # Store the feedback in your database or perform any other necessary actions
    print(feedback)
    feedback_vals = {}
    input_text = feedback.assessment_text
    feedback_vals['endoscopy'] = feedback.endoscopy
    feedback_vals['colonoscopy'] = feedback.colonoscopy
    feedback_vals['lumbar_puncture'] = feedback.lumbar_puncture
    feedback_vals['mri'] = feedback.mri
    feedback_vals['pet'] = feedback.pet
    feedback_vals['xray'] = feedback.xray
    
    idx = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    print(idx, input_text, str(feedback_vals))
    table_creation_logging.create_feedback()
    table_creation_logging.log_feedback(idx, input_text, str(feedback_vals), args.run_id)

    return {'message': 'Feedback collected successfully', 'feedback': feedback}

@app.post("/explain", response_model=ExplanationResult)
async def explain(input_text: Multilabel):
    run = mlflow.get_run(args.run_id)
    model_name = run.data.params['_name_or_path']
    model_run_name = run.data.tags['mlflow.runName']
    stage = 'Production' # config
    model_uri = f"models:/{model_run_name}/{stage}"
    model = mlflow.pytorch.load_model(model_uri)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    num_labels = len(model.config.id2label)
    explanation = generate_lime_explanation(model, tokenizer, input_text.assessment_text, num_labels)
    return {"explanation": explanation}

@app.post('/predict')
async def predict_species(multilabel_prompt: Multilabel):
    data = multilabel_prompt.dict()
    data = data['assessment_text']
    model_info = {
        "model_name": args.model_name,
        "experiment_name": args.experiment_name,
        "run_id": args.run_id,
        "port": args.port
    }
    predicted_labels_list, actual_labels_list, label_probs = prediction_multi_label.predict_service(args.run_id, data, model_info, stage='Production')

    return label_probs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host IP address")
    parser.add_argument("--port", type=int, default=8080, help="Port number")
    parser.add_argument("--mlflow_tracking_uri", type=str, help="MLflow tracking URI")
    args = parser.parse_args()

    model_path = args.run_id + "/" + args.model_name
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)

    if os.path.exists(model_path):
        model, tokenizer = load_model_and_tokenizer(model_path)
        print('MODEL NAME ------------------',model)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

