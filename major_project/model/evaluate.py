"""
Evaluation Script for Phishing URL Detection Model

Usage:
    python -m model.evaluate --dataset data/urls.csv
    
This script loads the active model (RF, DL, or Heuristic) and calculates
its accuracy, precision, recall, and f1-score against the provided dataset.
"""

import argparse
import pandas as pd
import numpy as np
import os
import sys
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.predict import PhishingPredictor

def load_testing_data(csv_path: str):
    print(f"\n Loading testing dataset from: {csv_path}")
    df = pd.read_csv(csv_path)

    url_col = None
    label_col = None
    for col in df.columns:
        if col.lower() in ("url", "urls", "uri", "link"):
            url_col = col
        if col.lower() in ("label", "class", "target", "phishing", "type", "result"):
            label_col = col

    if url_col is None or label_col is None:
        raise ValueError(f"Cannot detect columns. Found: {list(df.columns)}.")

    urls = df[url_col].astype(str).values
    labels = df[label_col].values

    unique_labels = set(labels)
    if unique_labels <= {"legitimate", "phishing"}:
        labels = np.array([1 if l == "phishing" else 0 for l in labels])
    elif unique_labels <= {"good", "bad"}:
        labels = np.array([1 if l == "bad" else 0 for l in labels])
    else:
        labels = labels.astype(int)
        
    return urls, labels

def evaluate_model(args):
    urls, y_true = load_testing_data(args.dataset)
    
    # Initialize the predictor (It will automatically load DL or RF or Heuristic)
    predictor = PhishingPredictor()
    
    if predictor.using_dl:
        active_model = "Deep Learning (CNN-BiLSTM)"
    elif predictor.using_rf:
        active_model = "Random Forest"
    else:
        active_model = "Heuristic Predictor"
        
    print(f" Checking Accuracy using active model: {active_model}...\n")
    
    y_pred = []
    
    print(f" Processing {len(urls)} URLs...")
    for i, url in enumerate(urls):
        result = predictor.predict(url)
        # Using 0.5 as threshold for phishing prediction
        pred_label = 1 if result["is_phishing"] else 0
        y_pred.append(pred_label)
        
        if (i + 1) % 1000 == 0:
            print(f"   Evaluated {i+1}/{len(urls)}")

    y_pred = np.array(y_pred)
    
    acc = accuracy_score(y_true, y_pred)
    print(f"\n=========================================")
    print(f" OVERALL ACCURACY : {acc * 100:.2f}%")
    print(f"=========================================\n")
    
    print(" Detailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Legitimate (0)", "Phishing (1)"]))
    
    print(" Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Model Accuracy against a Dataset")
    parser.add_argument("--dataset", type=str, required=True, help="Path to CSV dataset containing URLs and labels")
    
    args = parser.parse_args()
    evaluate_model(args)
