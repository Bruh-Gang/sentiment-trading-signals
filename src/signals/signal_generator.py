"""
Signal Generator
================
Converts sentiment scores into directional trade signals.
Supports thresholding, signal decay, and multi-ticker aggregation.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class SignalDirection(Enum):
    LONG  = "LONG"
    SHORT = "SHORT"
    FLAT  = "FLAT"


@dataclass
class TradeSignal:
    """A single directional trade signal."""
    ticker:     str
    direction:  SignalDirection
    score:      float      # Underlying sentiment score [0, 1]
    confidence: float      # Signal confidence [0, 1]
    timestamp:  datetime
    size:       float = 0.0       # Suggested position size (fraction of portfolio)
    ttl_hours:  int   = 24        # Time-to-live for signal decay
    source:     str   = "ensemble"

    @property
    def is_active(self) -> bool:
        """Return True if signal is still within TTL."""
        return datetime.utcnow() < self.timestamp + timedelta(hours=self.ttl_hours)

    @property
    def age_hours(self) -> float:
        return (datetime.utcnow() - self.timestamp).total_seconds() / 3600


class SignalGenerator:
    """
    Converts sentiment ensemble results into actionable trade signals.

    Signal Logic:
        normalized_score > long_threshold  → LONG
        normalized_score < short_threshold → SHORT
        else                               → FLAT

    Signal confidence gates which signals actually generate trades:
        confidence < min_confidence → suppress signal (FLAT)
    """

    def __init__(
        self,
        long_threshold:  float = 0.65,
        short_threshold: float = 0.35,
        min_confidence:  float = 0.60,
        signal_ttl:      int   = 24,    # Hours signal stays active
    ) -> None:
        self.long_threshold  = long_threshold
        self.short_threshold = short_threshold
        self.min_confidence  = min_confidence
        self.signal_ttl      = signal_ttl
        self._signal_history: List[TradeSignal] = []

    def generate(
        self,
        ticker:     str,
        score:      float,     # Normalized sentiment score [0, 1]
        confidence: float,
        timestamp:  datetime = None,
        position_size: float = 0.0,
    ) -> TradeSignal:
        """
        Generate a trade signal from a sentiment score.

        Args:
            ticker:        Stock ticker.
            score:         Normalized sentiment score [0, 1].
            confidence:    Prediction confidence [0, 1].
            timestamp:     Signal generation time (default: now).
            position_size: Suggested allocation fraction.

        Returns:
            TradeSignal with direction and metadata.
        """
        ts = timestamp or datetime.utcnow()

        # Confidence gate
        if confidence < self.min_confidence:
            direction = SignalDirection.FLAT
        elif score > self.long_threshold:
            direction = SignalDirection.LONG
        elif score < self.short_threshold:
            direction = SignalDirection.SHORT
        else:
            direction = SignalDirection.FLAT

        signal = TradeSignal(
            ticker=ticker,
            direction=direction,
            score=score,
            confidence=confidence,
            timestamp=ts,
            size=position_size,
            ttl_hours=self.signal_ttl,
        )
        self._signal_history.append(signal)
        return signal

    def generate_from_ensemble(self, ensemble_result) -> TradeSignal:
        """
        Generate signal from an EnsembleResult object.

        Args:
            ensemble_result: EnsembleResult from SentimentEnsemble.

        Returns:
            TradeSignal.
        """
        # Extract ticker from text (simplified — in production, pass ticker explicitly)
        ticker = getattr(ensemble_result, 'ticker', 'UNKNOWN')
        return self.generate(
            ticker=ticker,
            score=ensemble_result.normalized,
            confidence=ensemble_result.confidence,
        )

    def active_signals(self) -> List[TradeSignal]:
        """Return all signals that are still within TTL and non-FLAT."""
        return [s for s in self._signal_history
                if s.is_active and s.direction != SignalDirection.FLAT]

    def signal_to_numeric(self, signal: TradeSignal) -> float:
        """
        Convert signal direction to numeric target position.

        Returns:
            +1.0 for LONG, -1.0 for SHORT, 0.0 for FLAT
        """
        if signal.direction == SignalDirection.LONG:
            return +1.0
        elif signal.direction == SignalDirection.SHORT:
            return -1.0
        return 0.0

    def aggregate_signals(
        self, signals: List[TradeSignal], method: str = "mean"
    ) -> float:
        """
        Aggregate multiple signals for the same ticker.

        Args:
            signals: List of TradeSignal objects.
            method:  'mean', 'majority', or 'confidence_weighted'.

        Returns:
            Aggregate numeric signal [-1, 1].
        """
        if not signals:
            return 0.0

        numerics = [self.signal_to_numeric(s) for s in signals]

        if method == "mean":
            return sum(numerics) / len(numerics)
        elif method == "majority":
            pos = sum(1 for n in numerics if n > 0)
            neg = sum(1 for n in numerics if n < 0)
            if pos > neg: return 1.0
            if neg > pos: return -1.0
            return 0.0
        elif method == "confidence_weighted":
            total_conf = sum(s.confidence for s in signals)
            if total_conf == 0:
                return 0.0
            return sum(n * s.confidence for n, s in zip(numerics, signals)) / total_conf
        raise ValueError(f"Unknown method: {method}")

    def print_signals(self, signals: List[TradeSignal]) -> None:
        """Pretty-print a list of trade signals."""
        print(f"\n{'Ticker':<8} {'Direction':<8} {'Score':>7} {'Conf':>7} {'Size':>7} {'Age':>8}")
        print("-" * 55)
        for s in signals:
            print(f"{s.ticker:<8} {s.direction.value:<8} {s.score:>7.3f} "
                  f"{s.confidence:>7.3f} {s.size:>7.4f} {s.age_hours:>7.1f}h")


if __name__ == "__main__":
    gen = SignalGenerator()

    # Simulate some signals
    test_cases = [
        ("AAPL", 0.82, 0.91),
        ("TSLA", 0.18, 0.85),
        ("MSFT", 0.55, 0.72),
        ("NVDA", 0.71, 0.45),   # Low confidence — suppressed
        ("META", 0.30, 0.88),
    ]

    signals = []
    for ticker, score, conf in test_cases:
        s = gen.generate(ticker=ticker, score=score, confidence=conf)
        signals.append(s)

    gen.print_signals(signals)

    active = gen.active_signals()
    print(f"\nActive non-FLAT signals: {len(active)}")
    for s in active:
        print(f"  {s.ticker}: {s.direction.value}")
