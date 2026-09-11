"""
Training Pipeline for Random Forest Phishing URL Detector

Usage:
    python -m model.train_rf --dataset data/urls.csv
"""

import os
import sys
import argparse
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_text
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.feature_extractor import URLFeatureExtractor, FEATURE_NAMES

SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "saved_model")

def load_dataset(csv_path: str):
    """Load and validate dataset from CSV."""
    print(f"\n Loading dataset from: {csv_path}")
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

def preprocess(urls, labels, test_size=0.2):
    """Extract statistical features from URLs."""
    print("\n  Extracting features...")
    stat_features = []

    for i, url in enumerate(urls):
        _, stats = URLFeatureExtractor.extract_all(url)
        stat_features.append(stats)
        if (i + 1) % 5000 == 0:
            print(f"   Processed {i + 1}/{len(urls)}")

    stat_features = np.array(stat_features)

    X_train, X_test, y_train, y_test = train_test_split(
        stat_features, labels, test_size=test_size, random_state=42, stratify=labels
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler

def train(args):
    urls, labels = load_dataset(args.dataset)
    X_train, X_test, y_train, y_test, scaler = preprocess(urls, labels, test_size=args.test_size)

    print("\n Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth, 
        min_samples_split=args.min_samples_split,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)

    print("\n Evaluating on test set...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"]))
    
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    tree_rules = export_text(model.estimators_[0], feature_names=FEATURE_NAMES, max_depth=3)
    print("\n Random Forest Rules (Example Tree 0, Top 3 Depth):")
    print(tree_rules)

    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Save the Random Forest model
    rf_path = os.path.join(SAVE_DIR, "phishnet_rf.pkl")
    with open(rf_path, "wb") as f:
        pickle.dump(model, f)
        
    # Save scaler for RF
    scaler_path = os.path.join(SAVE_DIR, "scaler_rf.json")
    scaler_data = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "feature_names": FEATURE_NAMES,
    }
    with open(scaler_path, "w") as f:
        json.dump(scaler_data, f, indent=2)

    print(f"\n Random Forest model saved to {rf_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Random Forest for Phishing URL Detection")
    parser.add_argument("--dataset", type=str, required=True, help="Path to CSV dataset")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of trees in forest")
    parser.add_argument("--max-depth", type=int, default=10, help="Maximum depth of the tree")
    parser.add_argument("--min-samples-split", type=int, default=5, help="Minimum samples to split")
    
    args = parser.parse_args()
    train(args)
