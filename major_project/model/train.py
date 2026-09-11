"""
Training Pipeline for Phishing URL Detection Model

Usage:
    python -m model.train --dataset data/urls.csv --epochs 30 --batch-size 64

Dataset CSV format:
    url,label
    https://www.google.com,0
    http://phish.example.com/login.php,1
    ...
    (label: 0 = legitimate, 1 = phishing)
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.feature_extractor import URLFeatureExtractor, FEATURE_NAMES
from model.architecture import build_model, get_training_callbacks, model_summary_to_str

SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "saved_model")


def load_dataset(csv_path: str):
    """Load and validate dataset from CSV."""
    print(f"\n📂 Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Flexible column detection
    url_col = None
    label_col = None
    for col in df.columns:
        if col.lower() in ("url", "urls", "uri", "link"):
            url_col = col
        if col.lower() in ("label", "class", "target", "phishing", "type", "result"):
            label_col = col

    if url_col is None or label_col is None:
        raise ValueError(
            f"Cannot detect columns. Found: {list(df.columns)}. "
            "Need a URL column and a label column."
        )

    urls = df[url_col].astype(str).values
    labels = df[label_col].values

    # Convert labels to 0/1 if needed
    unique_labels = set(labels)
    if unique_labels <= {"legitimate", "phishing"}:
        labels = np.array([1 if l == "phishing" else 0 for l in labels])
    elif unique_labels <= {"good", "bad"}:
        labels = np.array([1 if l == "bad" else 0 for l in labels])
    else:
        labels = labels.astype(int)

    print(f"   Total samples : {len(urls)}")
    print(f"   Legitimate (0): {np.sum(labels == 0)}")
    print(f"   Phishing   (1): {np.sum(labels == 1)}")

    return urls, labels


def preprocess(urls, labels, test_size=0.2, val_size=0.1):
    """Extract features, split data, and scale."""
    print("\n⚙️  Extracting features...")
    char_encoded = []
    stat_features = []

    for i, url in enumerate(urls):
        chars, stats = URLFeatureExtractor.extract_all(url)
        char_encoded.append(chars)
        stat_features.append(stats)
        if (i + 1) % 5000 == 0:
            print(f"   Processed {i + 1}/{len(urls)}")

    char_encoded = np.array(char_encoded)
    stat_features = np.array(stat_features)

    print(f"   Character input shape : {char_encoded.shape}")
    print(f"   Statistical features  : {stat_features.shape}")

    # Train / val / test split
    X_char_train, X_char_test, X_stat_train, X_stat_test, y_train, y_test = (
        train_test_split(char_encoded, stat_features, labels, test_size=test_size, random_state=42, stratify=labels)
    )
    X_char_train, X_char_val, X_stat_train, X_stat_val, y_train, y_val = (
        train_test_split(X_char_train, X_stat_train, y_train, test_size=val_size, random_state=42, stratify=y_train)
    )

    # Scale statistical features
    scaler = StandardScaler()
    X_stat_train = scaler.fit_transform(X_stat_train)
    X_stat_val = scaler.transform(X_stat_val)
    X_stat_test = scaler.transform(X_stat_test)

    print(f"\n📊 Dataset split:")
    print(f"   Train : {len(y_train)}")
    print(f"   Val   : {len(y_val)}")
    print(f"   Test  : {len(y_test)}")

    return (
        (X_char_train, X_stat_train, y_train),
        (X_char_val, X_stat_val, y_val),
        (X_char_test, X_stat_test, y_test),
        scaler,
    )


def train(args):
    """Full training pipeline."""
    # Load data
    urls, labels = load_dataset(args.dataset)

    # Preprocess
    (X_char_train, X_stat_train, y_train), \
    (X_char_val, X_stat_val, y_val), \
    (X_char_test, X_stat_test, y_test), \
    scaler = preprocess(urls, labels, test_size=args.test_size)

    # Build model
    print("\n🏗️  Building model...")
    model = build_model(
        embedding_dim=args.embedding_dim,
        cnn_filters=args.cnn_filters,
        lstm_units=args.lstm_units,
        dense_units=args.dense_units,
        dropout_rate=args.dropout,
        l2_strength=args.l2,
        learning_rate=args.lr,
    )
    print(model_summary_to_str(model))

    # Compute class weights for imbalanced data (optimization technique)
    class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    print(f"\n⚖️  Class weights: {class_weight_dict}")

    # Callbacks
    cb = get_training_callbacks(SAVE_DIR)

    # Train
    print("\n🚀 Training started...")
    history = model.fit(
        [X_char_train, X_stat_train],
        y_train,
        validation_data=([X_char_val, X_stat_val], y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight_dict,
        callbacks=cb,
        verbose=1,
    )

    # Evaluate on test set
    print("\n📈 Evaluating on test set...")
    results = model.evaluate([X_char_test, X_stat_test], y_test, verbose=0)
    metrics = dict(zip(model.metrics_names, results))
    print(f"   Test metrics: {json.dumps(metrics, indent=2)}")

    # Classification report
    y_pred = (model.predict([X_char_test, X_stat_test]) > 0.5).astype(int).flatten()
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"]))

    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{cm}")

    # Save model and scaler
    os.makedirs(SAVE_DIR, exist_ok=True)
    model.save(os.path.join(SAVE_DIR, "phishnet_final.keras"))

    # Save scaler parameters
    scaler_data = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "feature_names": FEATURE_NAMES,
    }
    with open(os.path.join(SAVE_DIR, "scaler.json"), "w") as f:
        json.dump(scaler_data, f, indent=2)

    # Save training history
    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(os.path.join(SAVE_DIR, "history.json"), "w") as f:
        json.dump(hist, f, indent=2)

    print(f"\n✅ Model saved to {SAVE_DIR}/")
    print("   - phishnet_final.keras")
    print("   - phishnet_best.keras")
    print("   - scaler.json")
    print("   - history.json")

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PhishNet – Phishing URL Detector")
    parser.add_argument("--dataset", type=str, required=True, help="Path to CSV dataset")
    parser.add_argument("--epochs", type=int, default=30, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--dropout", type=float, default=0.4, help="Dropout rate")
    parser.add_argument("--l2", type=float, default=1e-4, help="L2 regularization strength")
    parser.add_argument("--embedding-dim", type=int, default=64, help="Character embedding dimension")
    parser.add_argument("--cnn-filters", type=int, default=128, help="CNN filter count")
    parser.add_argument("--lstm-units", type=int, default=64, help="LSTM hidden units")
    parser.add_argument("--dense-units", type=int, default=128, help="Dense layer units")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    args = parser.parse_args()

    train(args)
