import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
import yaml
from pathlib import Path

def load_dataset(path):
    project_root = Path(__file__).resolve().parents[2]
    training_path = project_root  / path / "training.csv"
    testing_path = project_root  / path / "testing.csv"
    validation_path = project_root  / path / "validation.csv"
    train_df = pd.read_csv(training_path)
    test_df = pd.read_csv(testing_path)
    validation_df = pd.read_csv(testing_path)
    train_df.columns = train_df.columns.str.lower()
    test_df.columns = test_df.columns.str.lower()
    validation_df.columns = validation_df.columns.str.lower()
    train = Dataset.from_pandas(train_df)
    test = Dataset.from_pandas(test_df)
    validation = Dataset.from_pandas(validation_df)
    return DatasetDict({'train': train, 'test': test, 'validation': validation})

def preprocess_data(examples, tokenizer, labels, config):
    text = examples["assessment_text"]
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=config['max_length'])
    labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
    labels_matrix = np.zeros((len(text), len(labels)))

    for idx, label in enumerate(labels):
        labels_matrix[:, idx] = labels_batch[label]

    encoding["labels"] = labels_matrix.tolist()

    return encoding
