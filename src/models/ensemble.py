"""
Ensemble Sentiment Model
=========================
Combines FinBERT and VADER predictions via weighted averaging.
Supports calibrated confidence thresholds and uncertainty estimation.
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EnsembleResult:
    """Result from the ensemble model."""
    text:           str
    ensemble_score: float   # ∈ [-1, 1]
    normalized:     float   # ∈ [0, 1]
    label:          str     # 'positive', 'negative', 'neutral'
    confidence:     float   # Confidence in label ∈ [0, 1]
    finbert_score:  Optional[float] = None
    vader_score:    Optional[float] = None


class SentimentEnsemble:
    """
    Weighted ensemble of FinBERT and VADER sentiment models.

    Configuration:
        finbert_weight: Weight for FinBERT predictions (default 0.70)
        vader_weight:   Weight for VADER predictions (default 0.30)
        Weights should sum to 1.0.

    The ensemble score is a weighted average of the two models'
    compound/normalized scores, both normalized to [-1, 1].
    """

    def __init__(
        self,
        finbert_weight: float = 0.70,
        vader_weight:   float = 0.30,
        long_threshold:  float = 0.65,   # Normalized score → LONG signal
        short_threshold: float = 0.35,   # Normalized score → SHORT signal
        min_confidence:  float = 0.60,
        use_mock_finbert: bool = False,   # Use mock if GPU/model unavailable
    ) -> None:
        assert abs(finbert_weight + vader_weight - 1.0) < 1e-6, "Weights must sum to 1"
        self.fw = finbert_weight
        self.vw = vader_weight
        self.long_threshold  = long_threshold
        self.short_threshold = short_threshold
        self.min_confidence  = min_confidence

        # Lazy-load models
        self._finbert = None
        self._vader   = None
        self._use_mock = use_mock_finbert

    def _get_finbert(self):
        if self._finbert is None:
            if self._use_mock:
                from src.models.finbert_sentiment import MockFinBERTSentiment
                self._finbert = MockFinBERTSentiment()
            else:
                from src.models.finbert_sentiment import FinBERTSentiment
                self._finbert = FinBERTSentiment()
        return self._finbert

    def _get_vader(self):
        if self._vader is None:
            from src.models.vader_baseline import VADERFinancial
            self._vader = VADERFinancial(extend_lexicon=True)
        return self._vader

    def _normalize(self, score: float) -> float:
        """Convert [-1, 1] compound score to [0, 1] normalized score."""
        return (score + 1.0) / 2.0

    def _label_from_normalized(self, normalized: float) -> Tuple[str, float]:
        """
        Determine label and confidence from normalized score.

        Returns:
            (label, confidence)
        """
        if normalized > self.long_threshold:
            label = "positive"
            confidence = (normalized - self.long_threshold) / (1.0 - self.long_threshold)
        elif normalized < self.short_threshold:
            label = "negative"
            confidence = (self.short_threshold - normalized) / self.short_threshold
        else:
            label = "neutral"
            # How close to center?
            confidence = 1.0 - 2 * abs(normalized - 0.5)

        return label, min(1.0, confidence)

    def predict(self, text: str) -> EnsembleResult:
        """
        Run ensemble prediction on a single text.

        Args:
            text: Financial news headline.

        Returns:
            EnsembleResult with combined scores and label.
        """
        finbert = self._get_finbert()
        vader   = self._get_vader()

        fb_score = finbert.sentiment_score(text)          # ∈ [-1, 1]
        vd_score = vader.compound_score(text)             # ∈ [-1, 1]

        # Weighted combination
        ensemble_score = self.fw * fb_score + self.vw * vd_score
        normalized     = self._normalize(ensemble_score)
        label, confidence = self._label_from_normalized(normalized)

        return EnsembleResult(
            text=text,
            ensemble_score=ensemble_score,
            normalized=normalized,
            label=label,
            confidence=confidence,
            finbert_score=fb_score,
            vader_score=vd_score,
        )

    def predict_batch(self, texts: List[str]) -> List[EnsembleResult]:
        """Batch prediction."""
        return [self.predict(t) for t in texts]

    def high_confidence_signals(
        self, texts: List[str]
    ) -> List[Tuple[str, EnsembleResult]]:
        """
        Return only high-confidence signals (confidence >= min_confidence).

        Args:
            texts: List of headlines.

        Returns:
            List of (text, result) for high-confidence predictions.
        """
        results = self.predict_batch(texts)
        return [
            (t, r) for t, r in zip(texts, results)
            if r.confidence >= self.min_confidence and r.label != "neutral"
        ]

    def aggregate_ticker_sentiment(
        self, ticker_headlines: Dict[str, List[str]]
    ) -> Dict[str, Dict]:
        """
        Aggregate sentiment across all headlines for each ticker.

        Args:
            ticker_headlines: Dict mapping ticker → list of headlines.

        Returns:
            Dict mapping ticker → aggregated stats.
        """
        aggregated = {}
        for ticker, headlines in ticker_headlines.items():
            if not headlines:
                continue
            results = self.predict_batch(headlines)
            scores  = [r.ensemble_score for r in results]
            pos = sum(1 for r in results if r.label == 'positive')
            neg = sum(1 for r in results if r.label == 'negative')
            aggregated[ticker] = {
                "mean_score":       sum(scores) / len(scores),
                "normalized_mean":  self._normalize(sum(scores) / len(scores)),
                "num_headlines":    len(headlines),
                "num_positive":     pos,
                "num_negative":     neg,
                "num_neutral":      len(results) - pos - neg,
                "dominant_label":   max(["positive", "negative", "neutral"],
                                         key=lambda l: sum(1 for r in results if r.label == l)),
            }
        return aggregated


if __name__ == "__main__":
    ensemble = SentimentEnsemble(use_mock_finbert=True)

    headlines = [
        "Apple surges 8% after blowout earnings, guidance raised",
        "Tesla crashes on massive miss, CEO sells shares",
        "Fed keeps rates unchanged, markets react calmly",
        "NVIDIA announces record data center revenue",
    ]

    print("=== Ensemble Predictions ===\n")
    for h in headlines:
        result = ensemble.predict(h)
        print(f"Text:       {h[:55]}...")
        print(f"Label:      {result.label:10}  Confidence: {result.confidence:.2f}")
        print(f"Score:      {result.ensemble_score:+.3f}  (FinBERT: {result.finbert_score:+.3f}, VADER: {result.vader_score:+.3f})")
        print()

    # Aggregate by ticker
    ticker_news = {
        "AAPL": ["Apple beats earnings", "Apple stock surges", "iPhone sales record"],
        "TSLA": ["Tesla misses estimates", "Tesla stock drops", "Musk sells shares"],
    }
    agg = ensemble.aggregate_ticker_sentiment(ticker_news)
    for ticker, stats in agg.items():
        print(f"{ticker}: {stats['dominant_label']} | mean={stats['mean_score']:+.3f} | n={stats['num_headlines']}")
