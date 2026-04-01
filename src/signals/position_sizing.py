"""
Position Sizing — Kelly Criterion and Variants
===============================================
Kelly Criterion determines the optimal fraction of portfolio to bet
on each signal to maximize long-run geometric growth.

Full Kelly can be extremely aggressive in practice; we use fractional Kelly.
"""

import math
import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def kelly_fraction(win_prob: float, win_loss_ratio: float) -> float:
    """
    Full Kelly Criterion for binary bet.

    Formula: f* = p - q/b
        where p = win probability
              q = 1 - p = loss probability
              b = win/loss ratio (net odds)

    Args:
        win_prob:       Probability of winning trade ∈ (0, 1).
        win_loss_ratio: Expected gain / expected loss.

    Returns:
        Optimal fraction of portfolio ∈ [0, 1].
        Returns 0 if Kelly recommends not betting.

    Example:
        >>> kelly_fraction(0.55, 1.5)
        0.1833   # Bet 18.3% of portfolio
    """
    q = 1.0 - win_prob
    f = win_prob - (q / win_loss_ratio)
    return max(0.0, f)


def fractional_kelly(
    win_prob: float,
    win_loss_ratio: float,
    fraction: float = 0.25,
    max_position: float = 0.10,
    min_position: float = 0.005,
) -> float:
    """
    Fractional Kelly position sizing (more conservative than full Kelly).

    In live trading, full Kelly leads to extreme drawdowns.
    25% Kelly (quarter-Kelly) is a common institutional approach.

    Args:
        win_prob:       P(trade wins).
        win_loss_ratio: Expected win / expected loss.
        fraction:       Kelly fraction multiplier (default 0.25).
        max_position:   Maximum allocation as fraction of portfolio.
        min_position:   Minimum allocation (below this → skip).

    Returns:
        Position size as fraction of portfolio.
    """
    full_kelly = kelly_fraction(win_prob, win_loss_ratio)
    size = full_kelly * fraction
    if size < min_position:
        return 0.0
    return min(size, max_position)


def confidence_based_sizing(
    confidence: float,
    sentiment_score: float,  # ∈ [0, 1], normalized
    base_size: float = 0.02,
    max_size:  float = 0.10,
) -> float:
    """
    Scale position size by both confidence and signal strength.

    size = base_size + (max_size - base_size) * confidence * |score - 0.5| * 2

    Args:
        confidence:      Model confidence [0, 1].
        sentiment_score: Normalized sentiment [0, 1].
        base_size:       Minimum position size.
        max_size:        Maximum position size.

    Returns:
        Position size ∈ [base_size, max_size].
    """
    signal_strength = abs(sentiment_score - 0.5) * 2  # Scale from center ∈ [0, 1]
    size = base_size + (max_size - base_size) * confidence * signal_strength
    return min(max_size, max(base_size, size))


def portfolio_level_sizing(
    signals: List[Dict],
    total_capital: float,
    max_gross_exposure: float = 1.5,   # 150% gross exposure max
    max_net_exposure:   float = 0.5,   # 50% net directional exposure
) -> List[Dict]:
    """
    Scale position sizes to satisfy portfolio-level constraints.

    Args:
        signals: List of dicts with 'ticker', 'direction' (+1/-1), 'raw_size'.
        total_capital: Total portfolio value.
        max_gross_exposure: Maximum sum of absolute positions.
        max_net_exposure:   Maximum net long/short.

    Returns:
        Signals with adjusted 'size' and 'dollar_value' fields.
    """
    gross = sum(s['raw_size'] for s in signals)
    net   = sum(s['raw_size'] * s['direction'] for s in signals)

    # Scale down if constraints violated
    scale_factor = 1.0
    if gross > max_gross_exposure:
        scale_factor = min(scale_factor, max_gross_exposure / gross)
    if abs(net) > max_net_exposure:
        scale_factor = min(scale_factor, max_net_exposure / abs(net))

    result = []
    for sig in signals:
        adj_size = sig['raw_size'] * scale_factor
        result.append({
            **sig,
            "size":        adj_size,
            "dollar_value": adj_size * total_capital * sig['direction'],
            "scale_factor": scale_factor,
        })
    return result


class KellyPositionSizer:
    """
    Stateful position sizer with rolling win/loss tracking.
    Estimates Kelly parameters from recent trade history.
    """

    def __init__(
        self,
        kelly_fraction:  float = 0.25,
        max_position:    float = 0.10,
        min_position:    float = 0.005,
        lookback_trades: int   = 50,
    ) -> None:
        self.kelly_frac   = kelly_fraction
        self.max_pos      = max_position
        self.min_pos      = min_position
        self.lookback     = lookback_trades
        self._trade_log: List[Dict] = []  # {'win': bool, 'pnl': float}

    def log_trade(self, won: bool, pnl: float) -> None:
        """Record a completed trade for parameter estimation."""
        self._trade_log.append({"win": won, "pnl": pnl})
        if len(self._trade_log) > self.lookback:
            self._trade_log.pop(0)

    def _estimate_params(self) -> Tuple[float, float]:
        """Estimate win probability and win/loss ratio from history."""
        if len(self._trade_log) < 5:
            return 0.52, 1.2  # Conservative defaults

        wins  = [t for t in self._trade_log if t["win"]]
        losses = [t for t in self._trade_log if not t["win"]]

        win_prob = len(wins) / len(self._trade_log)
        avg_win  = sum(abs(t["pnl"]) for t in wins)  / max(len(wins), 1)
        avg_loss = sum(abs(t["pnl"]) for t in losses) / max(len(losses), 1)
        ratio    = avg_win / max(avg_loss, 1e-9)

        return win_prob, ratio

    def compute_size(self, confidence: float = 1.0) -> float:
        """
        Compute position size using estimated Kelly parameters.

        Args:
            confidence: Scale by model confidence.

        Returns:
            Position size ∈ [0, max_position].
        """
        win_prob, ratio = self._estimate_params()
        size = fractional_kelly(win_prob, ratio, self.kelly_frac, self.max_pos, self.min_pos)
        return size * confidence


if __name__ == "__main__":
    # Simple Kelly examples
    print("=== Kelly Criterion Examples ===\n")
    examples = [
        (0.55, 1.5, "Slight edge"),
        (0.60, 2.0, "Good edge"),
        (0.45, 2.5, "Good odds, slight disadvantage"),
        (0.52, 1.1, "Typical algo edge"),
    ]
    for p, b, label in examples:
        full = kelly_fraction(p, b)
        frac = fractional_kelly(p, b, fraction=0.25)
        print(f"{label}: p={p}, b={b} | Full={full:.4f} | Quarter={frac:.4f}")

    # Confidence-based sizing
    print("\n=== Confidence-Based Sizing ===")
    for conf, score in [(0.95, 0.82), (0.75, 0.70), (0.60, 0.66)]:
        size = confidence_based_sizing(conf, score)
        print(f"conf={conf}, score={score} → size={size:.4f} ({size*100:.2f}%)")
