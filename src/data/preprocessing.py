"""
Text Preprocessing for Financial News
=======================================
Handles tokenization, cleaning, entity extraction, and feature engineering
for financial news headlines before sentiment modeling.

Requirements:
    pip install nltk spacy transformers
"""

import re
import string
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


# Financial-domain stopwords (keep words like "not", "up", "down" for sentiment)
FINANCIAL_STOPWORDS = {
    "the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or",
    "is", "are", "was", "were", "has", "have", "had", "be", "been", "being",
    "it", "its", "this", "that", "these", "those", "as", "with",
}

# Tickers and company name normalization
TICKER_PATTERN = re.compile(r'\b[A-Z]{1,5}\b')
URL_PATTERN    = re.compile(r'https?://\S+|www\.\S+')
HTML_PATTERN   = re.compile(r'<[^>]+>')
NUM_PATTERN    = re.compile(r'\$?[\d,]+\.?\d*[BMK]?%?')


class FinancialTextPreprocessor:
    """
    Preprocessing pipeline for financial news text.

    Steps:
        1. Lowercase (optional — FinBERT is case-sensitive, keep case for it)
        2. Remove URLs, HTML tags
        3. Normalize numbers and monetary values
        4. Sentence segmentation
        5. Ticker extraction
        6. Tokenization
    """

    def __init__(
        self,
        lowercase: bool = False,     # Keep case for transformer models
        remove_stopwords: bool = False,  # Keep for VADER context
        max_length: int = 512,
    ) -> None:
        self.lowercase        = lowercase
        self.remove_stopwords = remove_stopwords
        self.max_length       = max_length

    def clean_text(self, text: str) -> str:
        """
        Apply all cleaning steps to raw text.

        Args:
            text: Raw input string.

        Returns:
            Cleaned string.
        """
        # Remove HTML tags
        text = HTML_PATTERN.sub(' ', text)
        # Remove URLs
        text = URL_PATTERN.sub(' ', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove non-ASCII characters (keep punctuation for sentiment)
        text = text.encode('ascii', errors='ignore').decode('ascii')
        if self.lowercase:
            text = text.lower()
        return text

    def extract_tickers(self, text: str, known_tickers: Optional[List[str]] = None) -> List[str]:
        """
        Extract stock ticker symbols from text.

        Args:
            text:          Input text.
            known_tickers: Optional list of valid tickers to filter against.

        Returns:
            List of extracted ticker symbols.
        """
        candidates = TICKER_PATTERN.findall(text)
        if known_tickers:
            known_set = set(known_tickers)
            return [t for t in candidates if t in known_set]
        # Filter out common English words that look like tickers
        common_words = {"A", "I", "AT", "BE", "DO", "GO", "IS", "IT", "NO",
                        "OF", "ON", "OR", "TO", "UP", "US", "WE"}
        return [t for t in candidates if t not in common_words and len(t) >= 2]

    def normalize_numbers(self, text: str) -> str:
        """
        Normalize numeric expressions to token categories.
        e.g. "$1.2B" → "LARGE_POSITIVE_NUMBER", "down 15%" → "down PERCENTAGE"
        """
        # Keep percentage direction — important for sentiment
        text = re.sub(r'\$[\d,]+\.?\d*B', 'BILLION_DOLLARS', text)
        text = re.sub(r'\$[\d,]+\.?\d*M', 'MILLION_DOLLARS', text)
        text = re.sub(r'[\d,]+\.?\d*%',   'PERCENTAGE', text)
        text = re.sub(r'\$[\d,]+\.?\d*',  'DOLLAR_AMOUNT', text)
        return text

    def tokenize(self, text: str) -> List[str]:
        """
        Simple whitespace + punctuation tokenizer.
        For production, use transformers AutoTokenizer.

        Args:
            text: Cleaned text.

        Returns:
            List of tokens.
        """
        # Split on whitespace and punctuation (except apostrophes)
        tokens = re.findall(r"\b\w+(?:'\w+)?\b", text)
        if self.remove_stopwords:
            tokens = [t for t in tokens if t.lower() not in FINANCIAL_STOPWORDS]
        return tokens[:self.max_length]

    def preprocess(self, text: str) -> Dict:
        """
        Full preprocessing pipeline for a single text.

        Args:
            text: Raw input text.

        Returns:
            Dict with 'cleaned', 'tokens', 'tickers', 'num_tokens'.
        """
        cleaned = self.clean_text(text)
        tickers = self.extract_tickers(cleaned)
        tokens  = self.tokenize(cleaned)
        return {
            "original": text,
            "cleaned":  cleaned,
            "tokens":   tokens,
            "tickers":  tickers,
            "num_tokens": len(tokens),
        }

    def batch_preprocess(self, texts: List[str]) -> List[Dict]:
        """
        Preprocess a batch of texts.

        Args:
            texts: List of raw strings.

        Returns:
            List of preprocessed dicts.
        """
        return [self.preprocess(t) for t in texts]


def split_into_sentences(text: str) -> List[str]:
    """
    Simple sentence splitter for financial news.
    Handles abbreviations like "U.S.", "Corp.", "Inc."

    Args:
        text: Input text.

    Returns:
        List of sentences.
    """
    # Protect common abbreviations
    text = re.sub(r'(Mr|Mrs|Dr|Corp|Inc|Ltd|vs|U\.S|e\.g|i\.e)\.', r'\1<PERIOD>', text)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Restore periods
    sentences = [s.replace('<PERIOD>', '.') for s in sentences]
    return [s.strip() for s in sentences if s.strip()]


def compute_text_features(text: str) -> Dict:
    """
    Extract hand-crafted features from financial text (supplement to model features).

    Features:
        - num_positive_words: Count of positive financial words
        - num_negative_words: Count of negative financial words
        - has_number: Whether numeric values are present
        - exclamation_count: Number of exclamation marks
        - caps_ratio: Ratio of capital letters

    Args:
        text: Input text.

    Returns:
        Feature dictionary.
    """
    POSITIVE_WORDS = {
        "beat", "surge", "rally", "gain", "profit", "growth", "record",
        "boost", "rise", "strong", "upgrade", "buy", "outperform", "exceed",
        "positive", "increase", "higher", "improve", "recover",
    }
    NEGATIVE_WORDS = {
        "miss", "fall", "drop", "loss", "decline", "crash", "sell",
        "downgrade", "weak", "below", "cut", "reduce", "lower", "concern",
        "risk", "warning", "underperform", "decrease", "disappoint",
    }

    words = text.lower().split()
    pos = sum(1 for w in words if w.strip(string.punctuation) in POSITIVE_WORDS)
    neg = sum(1 for w in words if w.strip(string.punctuation) in NEGATIVE_WORDS)
    caps = sum(1 for c in text if c.isupper())

    return {
        "num_positive_words":  pos,
        "num_negative_words":  neg,
        "sentiment_word_diff": pos - neg,
        "has_number":          bool(NUM_PATTERN.search(text)),
        "exclamation_count":   text.count('!'),
        "caps_ratio":          caps / max(len(text), 1),
        "word_count":          len(words),
    }


if __name__ == "__main__":
    preprocessor = FinancialTextPreprocessor()
    samples = [
        "Apple Inc. (AAPL) beats Q3 earnings by $0.12, raises FY guidance",
        "Tesla TSLA stock drops 8% after Elon Musk sells $3.5B in shares",
        "Federal Reserve signals rate hike — markets tumble, S&P 500 falls 2.1%",
    ]
    for text in samples:
        result = preprocessor.preprocess(text)
        feats  = compute_text_features(text)
        print(f"\nOriginal: {result['original'][:60]}...")
        print(f"Tickers:  {result['tickers']}")
        print(f"Tokens:   {result['num_tokens']}")
        print(f"Features: pos={feats['num_positive_words']}, neg={feats['num_negative_words']}")
