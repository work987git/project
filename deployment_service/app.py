import os
import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from fastapi import FastAPI, Body, Request, status, HTTPException
from pydantic import BaseModel
import torch
from uuid import uuid4
import uvicorn
from lime.lime_text import LimeTextExplainer
from typing import List, Tuple
# from download_s3_dir import download_s3_directory
from monitoring import DatabaseManagement, APIMonitoring
from utils import download_artifacts, get_run_info
import mlflow
import sys
print(sys.path)
sys.path.append(r'C:\Users\U1153226\Documents\pb_pivot2\mlflow\clinical_trial_repo\clinical_project-main\clinical_prj\project')
from prediction_service import prediction_multi_label

class InputText(BaseModel):
    text: str

class OutputLabel(BaseModel):
    label: str
    score: float

class ExplanationResult(BaseModel):
    explanation: List[Tuple[str, float]]

def load_model_and_tokenizer(run_id: str):
    run = mlflow.get_run(run_id)
    model_name = run.data.params['_name_or_path']
    model_run_name = run.data.tags['mlflow.runName']
    stage = 'Production' # config
    model_uri = f"models:/{model_run_name}/{stage}"
    model = mlflow.pytorch.load_model(model_uri)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def generate_lime_explanation(
        model, 
        tokenizer, 
        text, 
        num_labels, 
        num_features=10, 
        num_samples=500):
    def predictor(texts):
        inputs = tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits)
        return probs.numpy()

    explainer = LimeTextExplainer(class_names=[model.config.id2label[i] for i in range(num_labels)])
    exp = explainer.explain_instance(
        text, predictor, 
        num_features=num_features, 
        top_labels=num_labels, 
        num_samples=num_samples)
    return exp.as_list()
    
class MyApp:
    def __init__(self, args):
        self.app = FastAPI()
        self.args = args
        
        self.app.on_event("startup")(self.startup_event)
        self.app.on_event("shutdown")(self.shutdown_event)

        self.app.post("/explain", response_model=ExplanationResult)(self.explain)
        self.app.post("/predict", response_model=list[OutputLabel])(self.predict)
        self.app.post("/submit_feedback")(self.submit_feedback)


        
        # model_path = download_artifacts(args.run_id, args.mlflow_tracking_uri)
        self.run_info = get_run_info(args.run_id, args.mlflow_tracking_uri)
        self.run_info["host"] =  self.args.host
        self.run_info["port"] =  self.args.port
        self.model, self.tokenizer = load_model_and_tokenizer(args.run_id)

    async def startup_event(self):
        
        api_monitoring.log_event("startup", self.run_info, "active")

    def shutdown_event(self):
        
        api_monitoring.log_event("shutdown", self.run_info, "inactive")

    async def explain(self, input_text: InputText):

        
        num_labels = len(self.model.config.id2label)
        explanation = generate_lime_explanation(
            self.model, 
            self.tokenizer, 
            input_text.text, 
            num_labels)
        
        return {"explanation": explanation}

    async def predict(
            self, 
            input_text: InputText, 
            request: Request):
        request_id = str(uuid4())
        try:
            input_data = await request.json()
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid JSON input")
        
        data = input_text.text
        # inputs = self.tokenizer(input_text.text, return_tensors="pt")
        # with torch.no_grad():
        #     outputs = self.model(**inputs)
        #     logits = outputs.logits
        #     probs = torch.sigmoid(logits)
        #     label_probs = [{'label': self.model.config.id2label[idx], 'score': prob.item()} for idx, prob in enumerate(probs[0])]
        
        predicted_labels_list, actual_labels_list, label_probs = prediction_multi_label.predict_service(args.run_id, data, stage='Production')

        api_monitoring.log_data(
            request_id, 
            input_data, 
            label_probs,  
            self.run_info)

        return label_probs

    class FeedbackData(BaseModel):
        query_text: str
        original_predictions: List[OutputLabel]
        run_id: str
        host: str
        port: int
        feedback_labels: List[str]
        timestamp: str

    async def submit_feedback(self, feedback_data: FeedbackData):
        api_monitoring.write_feedback_to_db(feedback_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host IP address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--mlflow_tracking_uri", type=str, help="MLflow tracking URI")
    parser.add_argument("--logging_db", type=str, help="DB used for logging")
    args = parser.parse_args()

    # Usage example
    db_management = DatabaseManagement(args.logging_db)
    api_monitoring = APIMonitoring(db_management)

    my_app = MyApp(args)

    uvicorn.run(
        my_app.app, 
        host=args.host, 
        port=args.port, 
        log_level="info")

