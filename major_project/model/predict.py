"""
Prediction Module for Phishing URL Detection

Loads a trained deep learning model or falls back to a heuristic-based
predictor so the web app works immediately without training.
"""

import os
import json
import pickle
import numpy as np

from .feature_extractor import (
    URLFeatureExtractor,
    FEATURE_NAMES,
    SUSPICIOUS_TLDS,
    SUSPICIOUS_KEYWORDS,
    IP_PATTERN,
    SHORTENING_DOMAINS,
)

SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "saved_model")


class HeuristicPredictor:
    """
    Rule-based phishing score when no trained model is available.
    Returns a probability-like score between 0 and 1.
    """

    # Feature weights tuned from common phishing indicators
    WEIGHTS = {
        "url_length": (0.008, 0.0),        # longer URLs = more suspicious
        "domain_length": (0.005, 0.0),
        "num_dots": (0.06, -0.06),          # many dots suspicious
        "num_hyphens": (0.08, 0.0),
        "num_underscores": (0.06, 0.0),
        "num_digits": (0.02, 0.0),
        "num_special_chars": (0.03, 0.0),
        "has_ip_address": (0.35, 0.0),      # IP in URL very suspicious
        "has_at_symbol": (0.30, 0.0),
        "is_https": (0.0, -0.15),           # HTTPS reduces suspicion
        "num_subdomains": (0.10, 0.0),
        "has_suspicious_tld": (0.25, 0.0),
        "url_entropy": (0.04, 0.0),
        "digit_ratio": (0.50, 0.0),
        "has_suspicious_keyword": (0.20, 0.0),
        "num_query_params": (0.04, 0.0),
        "path_depth": (0.04, 0.0),
        "has_port": (0.20, 0.0),
        "domain_digit_count": (0.05, 0.0),
        "has_redirect": (0.30, 0.0),
        "is_shortened": (0.15, 0.0),
    }

    @classmethod
    def predict(cls, url: str) -> float:
        """Return phishing probability (0–1) using heuristics."""
        features = URLFeatureExtractor.feature_dict(url)
        score = 0.0

        for name, (pos_w, neg_w) in cls.WEIGHTS.items():
            val = features.get(name, 0)
            if val > 0:
                score += val * pos_w
            else:
                score += abs(val) * neg_w

        # Clamp to [0, 1]
        score = max(0.0, min(1.0, score))
        return score


class PhishingPredictor:
    """
    Main predictor class. Uses the trained DL model if available,
    otherwise falls back to heuristic scoring.
    """

    def __init__(self):
        self.model = None
        self.rf_model = None
        self.scaler_mean = None
        self.scaler_scale = None
        
        self.using_dl = False
        self.using_rf = False
        self.using_heuristic = True

        self._try_load_model()

    def _try_load_model(self):
        """Attempt to load trained models (DL first, then RF)."""
        model_path = os.path.join(SAVE_DIR, "phishnet_best.keras")
        scaler_path = os.path.join(SAVE_DIR, "scaler.json")
        if not os.path.exists(model_path):
            model_path = os.path.join(SAVE_DIR, "phishnet_final.keras")

        # Try to load Deep Learning model
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
                from tensorflow import keras
                self.model = keras.models.load_model(model_path)
                with open(scaler_path) as f:
                    scaler_data = json.load(f)
                self.scaler_mean = np.array(scaler_data["mean"], dtype=np.float32)
                self.scaler_scale = np.array(scaler_data["scale"], dtype=np.float32)
                self.using_dl = True
                self.using_heuristic = False
                print(f" Loaded DL model from {model_path}")
            except Exception as e:
                print(f"  Could not load DL model: {e}")

        # Try to load Random Forest if DL is not used or as a parallel option
        rf_path = os.path.join(SAVE_DIR, "phishnet_rf.pkl")
        rf_scaler_path = os.path.join(SAVE_DIR, "scaler_rf.json")
        if not self.using_dl and os.path.exists(rf_path) and os.path.exists(rf_scaler_path):
             try:
                 with open(rf_path, "rb") as f:
                     self.rf_model = pickle.load(f)
                 with open(rf_scaler_path) as f:
                     rf_scaler_data = json.load(f)
                 self.scaler_mean = np.array(rf_scaler_data["mean"], dtype=np.float32)
                 self.scaler_scale = np.array(rf_scaler_data["scale"], dtype=np.float32)
                 self.using_rf = True
                 self.using_heuristic = False
                 print(f" Loaded Random Forest model from {rf_path}")
             except Exception as e:
                 print(f"  Could not load RF model: {e}")

        if not self.using_dl and not self.using_rf:
            print("  No trained model found. Using heuristic predictor.")

    def predict(self, url: str) -> dict:
        """
        Predict whether a URL is phishing.
        Returns a dict with prediction details.
        """
        features = URLFeatureExtractor.feature_dict(url)

        if self.using_heuristic:
            probability = HeuristicPredictor.predict(url)
        elif self.using_rf:
            char_enc, stat_feat = URLFeatureExtractor.extract_all(url)
            stat_feat = (stat_feat - self.scaler_mean) / self.scaler_scale
            probability = float(self.rf_model.predict_proba([stat_feat])[0][1])
        else:
            char_enc, stat_feat = URLFeatureExtractor.extract_all(url)
            # Scale statistical features
            stat_feat = (stat_feat - self.scaler_mean) / self.scaler_scale
            # Predict
            char_input = np.expand_dims(char_enc, axis=0)
            stat_input = np.expand_dims(stat_feat, axis=0)
            probability = float(self.model.predict(
                [char_input, stat_input], verbose=0
            )[0][0])

        # Determine risk level
        if probability >= 0.75:
            risk_level = "HIGH"
            verdict = "Phishing"
        elif probability >= 0.45:
            risk_level = "MEDIUM"
            verdict = "Suspicious"
        else:
            risk_level = "LOW"
            verdict = "Legitimate"

        # Top contributing features (for explainability)
        feature_importance = self._get_feature_importance(features)

        return {
            "url": url,
            "probability": round(probability, 4),
            "risk_level": risk_level,
            "verdict": verdict,
            "is_phishing": probability >= 0.5,
            "using_heuristic": self.using_heuristic,
            "using_rf": self.using_rf,
            "using_dl": self.using_dl,
            "features": features,
            "feature_importance": feature_importance,
        }

    @staticmethod
    def _get_feature_importance(features: dict) -> list:
        """
        Calculate which features contribute most to suspicion.
        Returns sorted list of (feature_name, impact_score) tuples.
        """
        importance = []

        risk_indicators = {
            "url_length": lambda v: min(v / 100, 1.0),
            "has_ip_address": lambda v: v,
            "has_at_symbol": lambda v: v,
            "has_suspicious_tld": lambda v: v,
            "has_suspicious_keyword": lambda v: v,
            "has_redirect": lambda v: v,
            "has_port": lambda v: v,
            "is_shortened": lambda v: v,
            "num_subdomains": lambda v: min(v / 4, 1.0),
            "num_dots": lambda v: min(v / 6, 1.0),
            "num_hyphens": lambda v: min(v / 4, 1.0),
            "digit_ratio": lambda v: min(v * 3, 1.0),
            "url_entropy": lambda v: min(v / 5, 1.0),
            "is_https": lambda v: 1.0 - v,  # NOT having https is risky
            "domain_digit_count": lambda v: min(v / 5, 1.0),
            "num_special_chars": lambda v: min(v / 10, 1.0),
            "path_depth": lambda v: min(v / 5, 1.0),
        }

        for name, scorer in risk_indicators.items():
            val = features.get(name, 0)
            score = scorer(val)
            if score > 0.01:
                importance.append({
                    "feature": name,
                    "value": val,
                    "impact": round(score, 3),
                })

        importance.sort(key=lambda x: x["impact"], reverse=True)
        return importance[:10]  # top 10
