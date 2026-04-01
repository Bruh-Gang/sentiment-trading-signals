"""
Vectorized Backtest Engine
===========================
Simulates portfolio P&L from sentiment signals and historical price data.
Uses vectorized NumPy/pandas operations for speed.

Input format:
    signals: DataFrame with columns [date, ticker, signal (+1/-1/0), size]
    prices:  DataFrame with index=date, columns=tickers (adjusted close prices)
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Container for backtest performance results."""
    returns:       pd.Series       # Daily strategy returns
    equity_curve:  pd.Series       # Cumulative equity value
    positions:     pd.DataFrame    # Position weights over time
    trades:        pd.DataFrame    # Trade log
    metrics:       Dict            # Performance metrics dict


class VectorizedBacktester:
    """
    Event-driven (but vectorized) backtester for sentiment signals.

    Assumptions:
        - Signals are executed at next-day open (realistic: no look-ahead)
        - Positions are held until a new signal or TTL expiry
        - Commission and slippage are applied on each trade

    Args:
        initial_capital: Starting portfolio value (USD).
        commission_bps:  Commission per trade in basis points (1 bps = 0.01%).
        slippage_bps:    Slippage per trade in basis points.
        signal_ttl:      Number of days to hold position without new signal.
    """

    def __init__(
        self,
        initial_capital: float = 100_000,
        commission_bps:  float = 5,
        slippage_bps:    float = 3,
        signal_ttl:      int   = 5,
        allow_short:     bool  = True,
    ) -> None:
        self.capital      = initial_capital
        self.comm_bps     = commission_bps / 10_000
        self.slip_bps     = slippage_bps   / 10_000
        self.ttl          = signal_ttl
        self.allow_short  = allow_short

    def run(
        self,
        signals: pd.DataFrame,
        prices:  pd.DataFrame,
    ) -> BacktestResult:
        """
        Execute backtest.

        Args:
            signals: DataFrame with columns ['date', 'ticker', 'signal', 'size'].
                     signal: +1 (long), -1 (short), 0 (flat).
                     size:   Fraction of portfolio to allocate.
            prices:  DataFrame indexed by date, columns = tickers.
                     Values = adjusted close prices.

        Returns:
            BacktestResult with full performance analysis.
        """
        prices = prices.sort_index()
        signals = signals.copy()
        signals['date'] = pd.to_datetime(signals['date'])
        signals = signals.sort_values('date')

        tickers = prices.columns.tolist()
        dates   = prices.index

        # Build position matrix: positions[date][ticker] = weight
        positions = pd.DataFrame(0.0, index=dates, columns=tickers)

        # Fill positions forward based on signals
        for _, row in signals.iterrows():
            ticker = row.get('ticker')
            if ticker not in tickers:
                continue
            signal = row.get('signal', 0)
            size   = row.get('size', 0.02)
            date   = row['date']
            if not self.allow_short and signal < 0:
                signal = 0.0

            # Find the first trading day on or after signal date
            future_dates = dates[dates >= date]
            if len(future_dates) == 0:
                continue
            start = future_dates[0]
            # Apply until TTL or end
            end_idx = dates.get_loc(start) + self.ttl
            end_idx = min(end_idx, len(dates))
            positions.loc[dates[dates.get_loc(start):end_idx], ticker] = signal * size

        # Calculate daily returns
        price_returns = prices.pct_change().fillna(0.0)

        # Strategy returns = sum(position * return) for each day
        # Positions are shifted forward by 1 day (signal at day t, execute at t+1)
        pos_shifted = positions.shift(1).fillna(0.0)
        strategy_returns = (pos_shifted * price_returns).sum(axis=1)

        # Apply transaction costs on position changes
        pos_changes = positions.diff().abs().sum(axis=1)
        costs = pos_changes * (self.comm_bps + self.slip_bps)
        strategy_returns -= costs

        # Equity curve
        equity = self.capital * (1 + strategy_returns).cumprod()

        # Build trade log
        trades = self._build_trade_log(positions, prices)

        from src.backtest.metrics import compute_metrics
        metrics = compute_metrics(strategy_returns, equity, self.capital)

        return BacktestResult(
            returns=strategy_returns,
            equity_curve=equity,
            positions=positions,
            trades=trades,
            metrics=metrics,
        )

    def _build_trade_log(
        self, positions: pd.DataFrame, prices: pd.DataFrame
    ) -> pd.DataFrame:
        """Extract entry/exit trades from position changes."""
        trades = []
        for ticker in positions.columns:
            pos = positions[ticker]
            changes = pos.diff().fillna(pos.iloc[0])
            for date in changes.index[changes != 0]:
                change = changes[date]
                if abs(change) < 1e-9:
                    continue
                price = prices.loc[date, ticker] if date in prices.index else None
                trades.append({
                    "date":      date,
                    "ticker":    ticker,
                    "direction": "LONG" if change > 0 else "SHORT" if change < 0 else "FLAT",
                    "size":      abs(change),
                    "price":     price,
                })
        return pd.DataFrame(trades)


def create_sample_backtest():
    """Generate synthetic signals and prices for demonstration."""
    np.random.seed(42)
    dates   = pd.date_range("2020-01-01", "2023-12-31", freq="B")
    tickers = ["AAPL", "MSFT", "GOOGL"]

    # Synthetic prices: random walk with upward drift
    prices = pd.DataFrame(index=dates, columns=tickers)
    for ticker in tickers:
        prices[ticker] = 100 * (1 + np.random.normal(0.0003, 0.015, len(dates))).cumprod()

    # Synthetic signals: random with slight positive bias
    signals = []
    for ticker in tickers:
        for i in range(0, len(dates), 5):
            date   = dates[i]
            signal = np.random.choice([1, -1, 0], p=[0.45, 0.30, 0.25])
            size   = np.random.uniform(0.01, 0.05)
            signals.append({"date": date, "ticker": ticker, "signal": signal, "size": size})

    signals_df = pd.DataFrame(signals)
    return signals_df, prices


if __name__ == "__main__":
    signals_df, prices_df = create_sample_backtest()
    engine = VectorizedBacktester(initial_capital=100_000, commission_bps=5)
    result = engine.run(signals_df, prices_df)

    print("=== Backtest Complete ===")
    print(f"Metrics: {result.metrics}")
    print(f"Final equity: ${result.equity_curve.iloc[-1]:,.2f}")
    print(f"Total trades: {len(result.trades)}")
