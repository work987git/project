import torch
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score
from transformers import EvalPrediction
import pickle
import yaml
from pathlib import Path
import mlflow
import os
from tqdm import tqdm

#save id2label and label2id to pickle file
def save_label_dict(model_save_dir, id2label, label2id):
    with open(model_save_dir + '/id2label.pkl', 'wb') as f:
        pickle.dump(id2label, f)
    with open(model_save_dir + '/label2id.pkl', 'wb') as f:
        pickle.dump(label2id, f)

def load_label_dict(model_save_dir):
    with open(model_save_dir / 'id2label.pkl', 'rb') as f:
        id2label = pickle.load(f)
    with open(model_save_dir / 'label2id.pkl', 'rb') as f:
        label2id = pickle.load(f)
    return id2label, label2id
    
def multi_label_metrics(predictions, labels):
    threshold=0.5
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    y_true = labels
    return {
        'f1': f1_score(y_true=y_true, y_pred=y_pred, average='micro'),
        'roc_auc': roc_auc_score(y_true, y_pred, average='micro'),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='micro')
        }

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    return multi_label_metrics(predictions=preds, labels=p.label_ids)

def compute_metrics_for_multiclass(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    true_start = p.label_ids[:, 0]
    true_end = p.label_ids[:, 1]

    pred_start = np.argmax(preds[:, 0], axis=1)
    pred_end = np.argmax(preds[:, 1], axis=1)

    exact_match = ((pred_start == true_start) & (pred_end == true_end)).astype(np.float32).mean().item()

    return {
        'exact_match': exact_match,
    }

def setup_mlflow(mlflow_config):

    # Set up the mlflow tracking server
    if mlflow_config['mlflow_artifact_uri'] is None:
        artifact_location = None
    else:
        artifact_location = mlflow_config['mlflow_artifact_uri']+"/"+mlflow_config['mlflow_experiment_name']

    mlflow.set_tracking_uri(mlflow_config['mlflow_tracking_uri'])

    try:
        experiment_id=mlflow.create_experiment(
            name=mlflow_config['mlflow_experiment_name'],
            artifact_location=artifact_location
        )
    except:
        mlflow.set_experiment(
            experiment_name=mlflow_config['mlflow_experiment_name'])
        experiment_id=mlflow.get_experiment_by_name(
            mlflow_config['mlflow_experiment_name']).experiment_id

    return experiment_id

class TqdmMLflowLogger:
    def __init__(self, tracking_uri=None):
        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)

    def log_artifacts(self, local_dir, artifact_path=None):
        local_dir = os.path.abspath(local_dir)
        artifact_files = []

        for root, _, files in os.walk(local_dir):
            artifact_files.extend([os.path.join(root, file) for file in files])

        with tqdm(total=len(artifact_files), desc="Uploading Artifacts") as pbar:
            for file in artifact_files:
                rel_path = os.path.relpath(file, local_dir)
                # Concatenate artifact_path and rel_path only if artifact_path is provided
                artifact_file_path = (
                    os.path.join(artifact_path, os.path.basename(rel_path)) if artifact_path else rel_path
                )
                # Replace backslashes with forward slashes in artifact_file_path
                artifact_file_path = artifact_file_path.replace("\\", "/")
                mlflow.log_artifact(file, artifact_path)
                pbar.update(1)





