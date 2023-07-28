import pandas as pd
import numpy as np

def preprocess_data(sample, tokenizer):
    text = sample["assessment_text"] if "assessment_text" in sample else sample["Assessment_text"]
    encoding = tokenizer(text, padding="max_length", 
                        truncation=True, 
                        max_length=512,
                        return_tensors = 'pt')

    return encoding