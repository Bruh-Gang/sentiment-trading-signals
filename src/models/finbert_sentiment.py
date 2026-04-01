"""
FinBERT Sentiment Model
========================
Uses the ProsusAI/finbert model from HuggingFace Transformers.
FinBERT is a pre-trained NLP model for analyzing financial sentiment.

Requirements:
    pip install transformers torch

Reference:
    Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-trained Language Models.
    https://arxiv.org/abs/1908.10063
"""

import logging
from typing import List, Dict, Union, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    """Sentiment analysis result for a single text."""
    text:       str
    label:      str        # 'positive', 'negative', 'neutral'
    score:      float      # Confidence score [0, 1]
    scores:     Dict[str, float] = None  # All class probabilities


class FinBERTSentiment:
    """
    Financial sentiment analysis using FinBERT (ProsusAI/finbert).

    Wraps the HuggingFace pipeline for convenient batch inference.
    Falls back gracefully if model is not available.

    Example:
        model = FinBERTSentiment()
        result = model.predict("Apple beats earnings by 12%")
        # SentimentResult(label='positive', score=0.97, ...)
    """

    MODEL_NAME = "ProsusAI/finbert"
    LABELS = ["positive", "negative", "neutral"]

    def __init__(
        self,
        model_name: str = None,
        device: int = -1,      # -1 = CPU, 0 = first GPU
        batch_size: int = 16,
        max_length: int = 512,
        cache_dir: str = None,
    ) -> None:
        self.model_name  = model_name or self.MODEL_NAME
        self.device      = device
        self.batch_size  = batch_size
        self.max_length  = max_length
        self.cache_dir   = cache_dir
        self._pipeline   = None

    def _load_pipeline(self):
        """Lazy-load the HuggingFace pipeline."""
        if self._pipeline is not None:
            return self._pipeline
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
            logger.info(f"Loading {self.model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir)
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, cache_dir=self.cache_dir
            )
            self._pipeline = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                device=self.device,
                top_k=None,       # Return all class scores
                truncation=True,
                max_length=self.max_length,
            )
            logger.info("FinBERT loaded successfully.")
        except ImportError:
            logger.error("transformers not installed. Install with: pip install transformers torch")
            raise
        return self._pipeline

    def predict(self, text: str) -> SentimentResult:
        """
        Predict sentiment for a single text.

        Args:
            text: Input financial news headline or article.

        Returns:
            SentimentResult with label and confidence score.
        """
        results = self.predict_batch([text])
        return results[0]

    def predict_batch(self, texts: List[str]) -> List[SentimentResult]:
        """
        Predict sentiment for a batch of texts.

        Args:
            texts: List of financial texts.

        Returns:
            List of SentimentResult objects.
        """
        pipe = self._load_pipeline()
        raw  = pipe(texts, batch_size=self.batch_size)

        results = []
        for text, output in zip(texts, raw):
            # output is a list of {label, score} dicts (top_k=None)
            scores_dict = {item['label'].lower(): item['score'] for item in output}
            best        = max(output, key=lambda x: x['score'])
            results.append(SentimentResult(
                text=text,
                label=best['label'].lower(),
                score=best['score'],
                scores=scores_dict,
            ))
        return results

    def sentiment_score(self, text: str) -> float:
        """
        Return a single float sentiment score in [-1, 1].

        Mapping:
            positive * score - negative * score
            neutral contributes 0

        Args:
            text: Input text.

        Returns:
            Float in [-1, 1] where 1 = very positive, -1 = very negative.
        """
        result = self.predict(text)
        if result.scores:
            return result.scores.get('positive', 0) - result.scores.get('negative', 0)
        if result.label == 'positive':
            return result.score
        elif result.label == 'negative':
            return -result.score
        return 0.0

    def batch_sentiment_scores(self, texts: List[str]) -> List[float]:
        """Return list of scalar sentiment scores for batch input."""
        results = self.predict_batch(texts)
        scores = []
        for r in results:
            if r.scores:
                scores.append(r.scores.get('positive', 0) - r.scores.get('negative', 0))
            elif r.label == 'positive':
                scores.append(r.score)
            elif r.label == 'negative':
                scores.append(-r.score)
            else:
                scores.append(0.0)
        return scores


class MockFinBERTSentiment:
    """
    Drop-in mock for FinBERT that uses simple keyword matching.
    Useful for testing the pipeline without loading the large model.
    """

    POSITIVE_WORDS = {
        "beat", "surge", "rally", "gain", "profit", "growth", "record",
        "boost", "rise", "upgrade", "buy", "outperform", "exceed",
    }
    NEGATIVE_WORDS = {
        "miss", "fall", "drop", "loss", "decline", "crash", "sell",
        "downgrade", "weak", "cut", "reduce", "concern", "warning",
    }

    def predict(self, text: str) -> SentimentResult:
        words  = set(text.lower().split())
        pos    = len(words & self.POSITIVE_WORDS)
        neg    = len(words & self.NEGATIVE_WORDS)
        total  = pos + neg + 1e-9
        if pos > neg:
            label = "positive"
            score = pos / total
        elif neg > pos:
            label = "negative"
            score = neg / total
        else:
            label = "neutral"
            score = 0.5
        return SentimentResult(
            text=text,
            label=label,
            score=score,
            scores={"positive": pos/total, "negative": neg/total, "neutral": 1/(total+1)},
        )

    def predict_batch(self, texts: List[str]) -> List[SentimentResult]:
        return [self.predict(t) for t in texts]

    def sentiment_score(self, text: str) -> float:
        r = self.predict(text)
        if r.label == 'positive': return r.score
        if r.label == 'negative': return -r.score
        return 0.0

    def batch_sentiment_scores(self, texts: List[str]) -> List[float]:
        return [self.sentiment_score(t) for t in texts]


if __name__ == "__main__":
    # Use mock for testing without downloading 400MB model
    model = MockFinBERTSentiment()
    headlines = [
        "Apple beats Q4 earnings estimates, stock surges 5%",
        "Tesla misses revenue targets, stock drops 8%",
        "Federal Reserve holds rates steady amid uncertainty",
        "NVIDIA reports record GPU sales, raises full-year guidance",
    ]
    print("=== FinBERT Sentiment Results ===\n")
    for h in headlines:
        result = model.predict(h)
        score  = model.sentiment_score(h)
        print(f"Text:  {h[:60]}...")
        print(f"Label: {result.label:8s}  Score: {score:+.3f}\n")
