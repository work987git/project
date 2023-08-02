import sys
from pathlib import Path
import yaml
import mlflow
import os
import logging
import mlflow.pytorch
import mlflow.pytorch
from transformers import AdamW

logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
from typing import Dict, Tuple
import argparse

import data_utils, model_utils

def load_config(config_path: str) -> Tuple[int, Dict[str, str]]:
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

class MlflowCallback(TrainerCallback):
    """Custom Callback for logging metrics to mlflow."""

    def on_epoch_end(self, args, state, control, metrics=None, **kwargs):
        if state.is_world_process_zero and metrics is not None:
            mlflow.log_metrics(metrics, step=state.epoch)         
def main(config_path: str):
    config = load_config(config_path)
    experiment_id = model_utils.setup_mlflow(config)
    print(experiment_id)
    dataset = data_utils.load_dataset(config['processed_data_path'])
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])

    labels = [label for label in dataset['train'].features.keys() if label not in [config["text_column"]]]
    id2label = {idx: label for idx, label in enumerate(labels)}
    label2id = {label: idx for idx, label in enumerate(labels)}

    encoded_dataset = dataset.map(
        lambda examples: data_utils.preprocess_data(examples, tokenizer, labels, config),
        batched=True,
        remove_columns=dataset['train'].column_names
    )

    encoded_dataset.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(
        config['model_name'],
        problem_type="multi_label_classification",
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id
    )

    model_path = config['model_save_dir']

    args = TrainingArguments(
        model_path,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config['learning_rate'],
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'],
        num_train_epochs=config['num_train_epochs'],
        weight_decay=config['weight_decay'],
        load_best_model_at_end=True,
        metric_for_best_model=config['metric_name'],
    )

    with mlflow.start_run(experiment_id=experiment_id):
        print("Artifact location: %s", mlflow.get_artifact_uri())
        print("Run %s" % mlflow.active_run().info.run_uuid)
        run_id = mlflow.active_run().info.run_uuid
        trainer = Trainer(
            model,
            args,
            train_dataset=encoded_dataset["train"],
            eval_dataset=encoded_dataset["validation"],
            tokenizer=tokenizer,
            compute_metrics=model_utils.compute_metrics,
            callbacks=[MlflowCallback]  
        )

        trainer.train()
        trainer.evaluate()
        profile_html_path = data_utils.create_and_log_profiling(config['processed_data_path'], run_id)
        # trainer.save_model(model_path)
        # mlflow.pytorch.save_model(model, model_path)
        mlflow.pytorch.log_model(trainer.model, "model")

        mlflow.log_artifacts(model_path, artifact_path="model")
        
        mlflow.log_artifact(profile_html_path, artifact_path="profiling_reports")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the configuration file",
    )
    args = parser.parse_args()
    main(args.config_path)