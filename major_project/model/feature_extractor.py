"""
URL Feature Extraction Module
Extracts statistical and character-level features from URLs for phishing detection.
"""

import re
import math
import string
from urllib.parse import urlparse, parse_qs
import numpy as np

# ---------------------------------------------------------------------------
# Character-level encoding constants
# ---------------------------------------------------------------------------
CHAR_VOCAB = (
    string.ascii_lowercase
    + string.digits
    + ".-/:_@?&=#%+~!$,;*'()|[]{}^`<>\"\\ \t"
)
CHAR_TO_IDX = {ch: idx + 1 for idx, ch in enumerate(CHAR_VOCAB)}  # 0 = padding
VOCAB_SIZE = len(CHAR_VOCAB) + 1  # +1 for the padding index
MAX_URL_LEN = 200

# ---------------------------------------------------------------------------
# Suspicious patterns
# ---------------------------------------------------------------------------
SUSPICIOUS_TLDS = {
    "zip", "review", "country", "kim", "cricket", "science",
    "work", "party", "gq", "link", "top", "ml", "cf", "ga",
    "tk", "xyz", "pw", "cc", "buzz", "click", "surf",
}

SUSPICIOUS_KEYWORDS = [
    "login", "verify", "secure", "account", "update", "confirm",
    "banking", "signin", "ebayisapi", "webscr", "paypal", "password",
    "credential", "suspend", "alert", "authenticate", "wallet",
    "recover", "unlock", "billing", "invoice", "document",
]

IP_PATTERN = re.compile(
    r"(?:\d{1,3}\.){3}\d{1,3}"  # simple IPv4 check
)

SHORTENING_DOMAINS = {
    "bit.ly", "goo.gl", "tinyurl.com", "ow.ly", "t.co",
    "is.gd", "buff.ly", "adf.ly", "cutt.ly", "rb.gy",
}

# List of feature names in the same order returned by extract()
FEATURE_NAMES = [
    "url_length",
    "domain_length",
    "path_length",
    "num_dots",
    "num_hyphens",
    "num_underscores",
    "num_slashes",
    "num_digits",
    "num_special_chars",
    "has_ip_address",
    "has_at_symbol",
    "is_https",
    "num_subdomains",
    "has_suspicious_tld",
    "url_entropy",
    "digit_ratio",
    "has_suspicious_keyword",
    "num_query_params",
    "path_depth",
    "has_port",
    "domain_digit_count",
    "has_redirect",
    "is_shortened",
    "longest_word_length",
    "avg_word_length",
]

NUM_FEATURES = len(FEATURE_NAMES)


class URLFeatureExtractor:
    """Extract both statistical features and character-level encoding from a URL."""

    @staticmethod
    def _safe_parse(url: str):
        """Parse URL, adding scheme if missing."""
        if not url.startswith(("http://", "https://")):
            url = "http://" + url
        return urlparse(url)

    @staticmethod
    def _entropy(text: str) -> float:
        """Shannon entropy of a string."""
        if not text:
            return 0.0
        freq = {}
        for ch in text:
            freq[ch] = freq.get(ch, 0) + 1
        length = len(text)
        return -sum(
            (count / length) * math.log2(count / length)
            for count in freq.values()
        )

    @classmethod
    def extract(cls, url: str) -> np.ndarray:
        """Return a 1-D numpy array of statistical features for *url*."""
        parsed = cls._safe_parse(url)
        hostname = parsed.hostname or ""
        path = parsed.path or ""
        query = parsed.query or ""

        # Basic lengths
        url_length = len(url)
        domain_length = len(hostname)
        path_length = len(path)

        # Character counts
        num_dots = url.count(".")
        num_hyphens = url.count("-")
        num_underscores = url.count("_")
        num_slashes = url.count("/")
        num_digits = sum(c.isdigit() for c in url)
        num_special = sum(c in "!@#$%^&*()+={}[]|\\:;'<>,?/~`" for c in url)

        # Boolean-ish flags (0/1)
        has_ip = 1 if IP_PATTERN.search(hostname) else 0
        has_at = 1 if "@" in url else 0
        is_https = 1 if parsed.scheme == "https" else 0

        # Subdomain depth
        parts = hostname.split(".")
        num_subdomains = max(len(parts) - 2, 0)

        # TLD check
        tld = parts[-1] if parts else ""
        has_suspicious_tld = 1 if tld.lower() in SUSPICIOUS_TLDS else 0

        # Entropy
        url_entropy = cls._entropy(url)

        # Digit ratio
        digit_ratio = num_digits / max(url_length, 1)

        # Suspicious keywords
        url_lower = url.lower()
        has_suspicious_kw = 1 if any(kw in url_lower for kw in SUSPICIOUS_KEYWORDS) else 0

        # Query params
        num_params = len(parse_qs(query))

        # Path depth
        path_depth = path.strip("/").count("/") + (1 if path.strip("/") else 0)

        # Port
        has_port = 1 if parsed.port and parsed.port not in (80, 443) else 0

        # Digits in domain
        domain_digit_count = sum(c.isdigit() for c in hostname)

        # Redirect pattern (double slashes after scheme)
        has_redirect = 1 if url.count("//") > 1 else 0

        # URL shortener
        is_shortened = 1 if any(sd in hostname for sd in SHORTENING_DOMAINS) else 0

        # Word-level stats from domain
        words = re.split(r"[.\-_/]", url)
        word_lens = [len(w) for w in words if w]
        longest_word = max(word_lens) if word_lens else 0
        avg_word = sum(word_lens) / len(word_lens) if word_lens else 0

        features = np.array(
            [
                url_length,
                domain_length,
                path_length,
                num_dots,
                num_hyphens,
                num_underscores,
                num_slashes,
                num_digits,
                num_special,
                has_ip,
                has_at,
                is_https,
                num_subdomains,
                has_suspicious_tld,
                url_entropy,
                digit_ratio,
                has_suspicious_kw,
                num_params,
                path_depth,
                has_port,
                domain_digit_count,
                has_redirect,
                is_shortened,
                longest_word,
                avg_word,
            ],
            dtype=np.float32,
        )
        return features

    @staticmethod
    def encode_url(url: str) -> np.ndarray:
        """Character-level integer encoding (padded/truncated to MAX_URL_LEN)."""
        encoded = [CHAR_TO_IDX.get(ch.lower(), 0) for ch in url[:MAX_URL_LEN]]
        if len(encoded) < MAX_URL_LEN:
            encoded += [0] * (MAX_URL_LEN - len(encoded))
        return np.array(encoded, dtype=np.int32)

    @classmethod
    def extract_all(cls, url: str):
        """Return (char_encoded, stat_features) for a single URL."""
        return cls.encode_url(url), cls.extract(url)

    @classmethod
    def feature_dict(cls, url: str) -> dict:
        """Return a dict mapping feature name → value."""
        values = cls.extract(url)
        return dict(zip(FEATURE_NAMES, values.tolist()))
