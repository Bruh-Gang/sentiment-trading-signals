"""
VADER Sentiment Baseline
========================
Valence Aware Dictionary and sEntiment Reasoner (VADER) is a rule-based
sentiment analysis tool designed for social media and short texts.

We extend it with a financial domain lexicon.

Requirements:
    pip install nltk vaderSentiment
"""

import re
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# Extended financial domain lexicon for VADER
# Format: {word: valence_score}  where score ∈ [-4.0, 4.0]
FINANCIAL_LEXICON: Dict[str, float] = {
    # Strongly positive
    "beat":        3.0,
    "beats":       3.0,
    "surge":       3.2,
    "surges":      3.2,
    "record":      2.5,
    "boom":        3.0,
    "outperform":  2.8,
    "upgrade":     2.5,
    "raised":      2.0,
    "raised guidance": 3.0,
    "dividend increase": 2.5,
    "buyback":     2.0,
    "acquisition": 1.5,

    # Positive
    "gain":        2.0,
    "rally":       2.3,
    "profit":      2.0,
    "growth":      2.0,
    "expand":      1.8,
    "recover":     1.8,
    "strong":      1.5,
    "robust":      1.5,
    "optimistic":  1.8,
    "bullish":     2.5,

    # Negative
    "miss":       -2.5,
    "misses":     -2.5,
    "drop":       -2.0,
    "drops":      -2.0,
    "fall":       -1.8,
    "decline":    -1.8,
    "loss":       -2.0,
    "risk":       -1.2,
    "concern":    -1.5,
    "weak":       -1.8,
    "bearish":    -2.5,
    "downgrade":  -2.5,
    "cut":        -1.8,
    "layoff":     -2.0,
    "layoffs":    -2.0,

    # Strongly negative
    "crash":      -3.5,
    "fraud":      -4.0,
    "bankrupt":   -4.0,
    "bankruptcy": -4.0,
    "collapse":   -3.8,
    "plunge":     -3.2,
    "plunges":    -3.2,
    "scandal":    -3.5,
    "lawsuit":    -2.5,
    "investigation": -2.8,
    "default":    -3.5,
}


class VADERFinancial:
    """
    Extended VADER sentiment analyzer with financial domain lexicon.

    Falls back to pure keyword scoring if NLTK is not available.
    """

    def __init__(self, extend_lexicon: bool = True) -> None:
        self._analyzer = None
        self.extended  = extend_lexicon
        self._load_vader()

    def _load_vader(self) -> None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._analyzer = SentimentIntensityAnalyzer()
            if self.extended:
                self._analyzer.lexicon.update(FINANCIAL_LEXICON)
                logger.info(f"Loaded VADER + {len(FINANCIAL_LEXICON)} financial terms")
        except ImportError:
            logger.warning("vaderSentiment not installed. Using fallback keyword scorer.")
            self._analyzer = None

    def polarity_scores(self, text: str) -> Dict[str, float]:
        """
        Return VADER compound and component scores.

        Returns:
            Dict with 'neg', 'neu', 'pos', 'compound' keys.
            compound ∈ [-1, 1], component scores sum to 1.
        """
        if self._analyzer is not None:
            return self._analyzer.polarity_scores(text)
        return self._fallback_score(text)

    def _fallback_score(self, text: str) -> Dict[str, float]:
        """Keyword-based fallback when VADER is not installed."""
        words = re.findall(r'\b\w+\b', text.lower())
        score = 0.0
        for word in words:
            score += FINANCIAL_LEXICON.get(word, 0.0)
        # Normalize to [-1, 1]
        compound = max(-1.0, min(1.0, score / (len(words) + 1e-9)))
        pos = max(0.0, compound)
        neg = max(0.0, -compound)
        neu = 1.0 - pos - neg
        return {"neg": neg, "neu": neu, "pos": pos, "compound": compound}

    def sentiment_label(self, text: str) -> str:
        """
        Return sentiment label based on compound score threshold.

        Thresholds (VADER standard):
            compound >= 0.05  → positive
            compound <= -0.05 → negative
            else              → neutral
        """
        scores = self.polarity_scores(text)
        c = scores["compound"]
        if c >= 0.05:
            return "positive"
        elif c <= -0.05:
            return "negative"
        return "neutral"

    def compound_score(self, text: str) -> float:
        """Return just the compound score ∈ [-1, 1]."""
        return self.polarity_scores(text)["compound"]

    def batch_compound_scores(self, texts: List[str]) -> List[float]:
        """Batch-process texts and return compound scores."""
        return [self.compound_score(t) for t in texts]

    def compound_to_normalized(self, compound: float) -> float:
        """
        Convert compound score from [-1, 1] to [0, 1].
        0 = very negative, 0.5 = neutral, 1 = very positive.
        """
        return (compound + 1.0) / 2.0


if __name__ == "__main__":
    vader = VADERFinancial(extend_lexicon=True)
    headlines = [
        "Apple surges 8% after record-breaking quarterly earnings beat",
        "Tesla stock crashes 15% following massive miss and guidance cut",
        "Federal Reserve holds interest rates steady for third consecutive meeting",
        "Goldman Sachs upgrades NVDA to Buy, sets $700 price target",
        "Regional bank faces fraud investigation, shares plunge 40%",
    ]
    print("=== VADER Financial Sentiment ===\n")
    print(f"{'Headline':<55} {'Label':10} {'Score':>8}")
    print("-" * 80)
    for h in headlines:
        label = vader.sentiment_label(h)
        score = vader.compound_score(h)
        print(f"{h[:55]:<55} {label:10} {score:>+8.3f}")
