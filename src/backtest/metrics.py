"""
Performance Metrics
====================
Sharpe ratio, Sortino ratio, maximum drawdown, win rate, and more.
All standard quant finance performance metrics for strategy evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


TRADING_DAYS_PER_YEAR = 252


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.04,
    annualize: bool = True,
) -> float:
    """
    Compute the Sharpe ratio.

    Formula: (mean_return - Rf) / std_return * sqrt(periods_per_year)

    Args:
        returns:        Daily returns series.
        risk_free_rate: Annual risk-free rate (default 4% for 2024).
        annualize:      Whether to annualize (default True).

    Returns:
        Sharpe ratio (annualized if annualize=True).
    """
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    daily_rf = (1 + risk_free_rate) ** (1 / TRADING_DAYS_PER_YEAR) - 1
    excess   = returns - daily_rf
    ratio    = excess.mean() / excess.std()
    if annualize:
        ratio *= np.sqrt(TRADING_DAYS_PER_YEAR)
    return ratio


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.04,
    target_return:  float = 0.0,
) -> float:
    """
    Sortino ratio — like Sharpe but penalizes only downside volatility.

    Formula: (mean_return - Rf) / downside_std * sqrt(252)

    Args:
        returns:        Daily returns.
        risk_free_rate: Annual risk-free rate.
        target_return:  Minimum acceptable return (MAR).

    Returns:
        Sortino ratio (annualized).
    """
    daily_rf  = (1 + risk_free_rate) ** (1 / TRADING_DAYS_PER_YEAR) - 1
    excess    = returns - daily_rf
    downside  = returns[returns < target_return]
    if len(downside) == 0 or downside.std() == 0:
        return float('inf')
    downside_std = downside.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    return (excess.mean() * TRADING_DAYS_PER_YEAR) / downside_std


def calmar_ratio(
    returns: pd.Series,
    equity:  pd.Series,
) -> float:
    """
    Calmar ratio = Annualized Return / Max Drawdown.

    Args:
        returns: Daily returns.
        equity:  Equity curve series.

    Returns:
        Calmar ratio.
    """
    ann_return = annualized_return(returns)
    mdd = max_drawdown(equity)
    if mdd == 0:
        return float('inf')
    return ann_return / abs(mdd)


def max_drawdown(equity: pd.Series) -> float:
    """
    Maximum peak-to-trough drawdown as a negative fraction.

    Args:
        equity: Equity curve (cumulative values, not returns).

    Returns:
        Maximum drawdown as negative fraction (e.g., -0.15 = -15%).
    """
    if len(equity) == 0:
        return 0.0
    rolling_max = equity.expanding().max()
    drawdown    = (equity - rolling_max) / rolling_max
    return drawdown.min()


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Return the full drawdown time series."""
    rolling_max = equity.expanding().max()
    return (equity - rolling_max) / rolling_max


def annualized_return(returns: pd.Series) -> float:
    """
    Compute compound annualized growth rate (CAGR).

    Args:
        returns: Daily return series.

    Returns:
        CAGR as decimal (e.g., 0.18 = 18%).
    """
    if len(returns) == 0:
        return 0.0
    total = (1 + returns).prod()
    n_years = len(returns) / TRADING_DAYS_PER_YEAR
    return total ** (1 / n_years) - 1


def win_rate(returns: pd.Series) -> float:
    """Fraction of trading days with positive returns."""
    if len(returns) == 0:
        return 0.0
    return (returns > 0).sum() / len(returns)


def profit_factor(returns: pd.Series) -> float:
    """
    Ratio of gross profits to gross losses.
    > 1 = profitable strategy.
    """
    gains  = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    if losses == 0:
        return float('inf')
    return gains / losses


def avg_win_loss_ratio(returns: pd.Series) -> float:
    """Average gain on winning days / average loss on losing days."""
    wins   = returns[returns > 0]
    losses = returns[returns < 0]
    avg_w  = wins.mean()   if len(wins)   > 0 else 0
    avg_l  = abs(losses.mean()) if len(losses) > 0 else 1e-9
    return avg_w / avg_l


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Historical VaR at given confidence level.

    Args:
        returns:    Daily returns.
        confidence: Confidence level (default 95%).

    Returns:
        VaR as negative fraction (worst loss at confidence level).
    """
    return np.percentile(returns, (1 - confidence) * 100)


def compute_metrics(
    returns: pd.Series,
    equity:  pd.Series,
    initial_capital: float = 100_000,
    risk_free_rate: float = 0.04,
) -> Dict:
    """
    Compute all standard performance metrics and return as dict.

    Args:
        returns:         Daily return series.
        equity:          Equity curve series.
        initial_capital: Starting capital.
        risk_free_rate:  Annual risk-free rate.

    Returns:
        Dict with all performance statistics.
    """
    non_zero = returns[returns != 0]
    return {
        "annualized_return":  annualized_return(returns),
        "total_return":       (equity.iloc[-1] / initial_capital - 1) if len(equity) > 0 else 0,
        "sharpe_ratio":       sharpe_ratio(returns, risk_free_rate),
        "sortino_ratio":      sortino_ratio(returns, risk_free_rate),
        "calmar_ratio":       calmar_ratio(returns, equity),
        "max_drawdown":       max_drawdown(equity),
        "win_rate":           win_rate(non_zero),
        "profit_factor":      profit_factor(non_zero),
        "avg_win_loss_ratio": avg_win_loss_ratio(non_zero),
        "var_95":             value_at_risk(returns, 0.95),
        "volatility_annual":  returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR),
        "num_trading_days":   len(returns),
    }


def print_metrics(metrics: Dict) -> None:
    """Pretty-print performance metrics."""
    print("\n" + "=" * 50)
    print("           PERFORMANCE METRICS")
    print("=" * 50)
    fmt = {
        "annualized_return":  "{:+.1%}",
        "total_return":       "{:+.1%}",
        "sharpe_ratio":       "{:.3f}",
        "sortino_ratio":      "{:.3f}",
        "calmar_ratio":       "{:.3f}",
        "max_drawdown":       "{:.1%}",
        "win_rate":           "{:.1%}",
        "profit_factor":      "{:.3f}",
        "avg_win_loss_ratio": "{:.3f}",
        "var_95":             "{:.3%}",
        "volatility_annual":  "{:.1%}",
        "num_trading_days":   "{:.0f}",
    }
    labels = {
        "annualized_return": "Annualized Return (CAGR)",
        "total_return":      "Total Return",
        "sharpe_ratio":      "Sharpe Ratio",
        "sortino_ratio":     "Sortino Ratio",
        "calmar_ratio":      "Calmar Ratio",
        "max_drawdown":      "Max Drawdown",
        "win_rate":          "Win Rate",
        "profit_factor":     "Profit Factor",
        "avg_win_loss_ratio":"Avg Win/Loss Ratio",
        "var_95":            "VaR (95%)",
        "volatility_annual": "Annual Volatility",
        "num_trading_days":  "Trading Days",
    }
    for key, val in metrics.items():
        label  = labels.get(key, key)
        fmtstr = fmt.get(key, "{}")
        print(f"  {label:<28}: {fmtstr.format(val)}")
    print("=" * 50)


if __name__ == "__main__":
    np.random.seed(0)
    dates   = pd.date_range("2020-01-01", "2023-12-31", freq="B")
    returns = pd.Series(np.random.normal(0.0005, 0.012, len(dates)), index=dates)
    equity  = 100_000 * (1 + returns).cumprod()

    metrics = compute_metrics(returns, equity)
    print_metrics(metrics)
