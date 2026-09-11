"""
Flask Web Server for Phishing URL Detection
Serves the frontend and exposes prediction API endpoints.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from model.predict import PhishingPredictor
from model.feature_extractor import FEATURE_NAMES

app = Flask(__name__)
CORS(app)

# Initialize predictor once at startup
predictor = PhishingPredictor()


@app.route("/")
def index():
    """Serve the main web page."""
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Predict phishing probability for a single URL.
    Expects JSON body: { "url": "https://example.com" }
    """
    data = request.get_json(force=True)
    url = data.get("url", "").strip()

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    if len(url) > 2048:
        return jsonify({"error": "URL too long (max 2048 characters)"}), 400

    try:
        result = predictor.predict(url)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/batch", methods=["POST"])
def batch_predict():
    """
    Predict phishing probability for multiple URLs.
    Expects JSON body: { "urls": ["url1", "url2", ...] }
    """
    data = request.get_json(force=True)
    urls = data.get("urls", [])

    if not urls:
        return jsonify({"error": "No URLs provided"}), 400

    if len(urls) > 50:
        return jsonify({"error": "Maximum 50 URLs per batch"}), 400

    results = []
    for url in urls:
        url = url.strip()
        if url:
            try:
                result = predictor.predict(url)
                results.append(result)
            except Exception as e:
                results.append({"url": url, "error": str(e)})

    return jsonify({"results": results})


@app.route("/api/features", methods=["POST"])
def get_features():
    """
    Return extracted features for a URL (for educational/debug purposes).
    """
    data = request.get_json(force=True)
    url = data.get("url", "").strip()

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    from model.feature_extractor import URLFeatureExtractor
    features = URLFeatureExtractor.feature_dict(url)
    return jsonify({"url": url, "features": features, "feature_names": FEATURE_NAMES})


@app.route("/api/status")
def status():
    """Return system status."""
    if predictor.using_dl:
        model_type = "deep_learning"
        model_name = "PhishNet CNN-BiLSTM"
    elif predictor.using_rf:
        model_type = "random_forest"
        model_name = "PhishNet Random Forest"
    else:
        model_type = "heuristic"
        model_name = "Heuristic (no trained model)"

    return jsonify({
        "status": "online",
        "model_type": model_type,
        "model_name": model_name,
        "feature_count": len(FEATURE_NAMES),
    })


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  🛡️  PhishNet – Phishing URL Detection System")
    print("=" * 60)
    
    if predictor.using_dl:
        model_str = "Deep Learning (CNN-BiLSTM)"
    elif predictor.using_rf:
        model_str = "Random Forest"
    else:
        model_str = "Heuristic (no trained model)"

    print(f"  Model: {model_str}")
    print(f"  Features: {len(FEATURE_NAMES)}")
    print(f"  Server: http://127.0.0.1:5000")
    print("=" * 60 + "\n")

    app.run(debug=True, host="0.0.0.0", port=5000)
