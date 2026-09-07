"""
Vectorized backtesting and performance analytics for quantitative strategies.

(c) Dr. Yves J. Hilpisch
The Python Quants GmbH | https://tpq.io
https://hilpisch.com
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import FX_COST_ONE_WAY


def run_vectorized_backtest(
    returns: pd.Series,
    positions: pd.Series,
    tc: float = FX_COST_ONE_WAY,
    periods_per_year: int = 252,
    lag_positions: bool = False
) -> tuple[pd.DataFrame, dict[str, float]]:
    r"""Execute a vectorized backtest with transaction costs.

    Convention:
      - `returns` at index t represents the log-return realized during bar t
        (from close t-1 to close t).
      - `positions` at index t represents the position held during bar t.
      - If `lag_positions=True`, `positions` is shifted by 1 bar to convert
        bar-end signals into next-bar holding positions.

    Parameters
    ----------
    returns : pd.Series
        Asset log-returns $r_t = \ln(P_t / P_{t-1})$.
    positions : pd.Series
        Target positions $p_t \in \{-1, 0, 1\}$.
    tc : float
        One-way proportional cost per unit turnover. A direct reversal has
        turnover of two and therefore incurs twice this cost.
    periods_per_year : int
        Annualization factor (252 for daily trading).
    lag_positions : bool
        If True, shifts positions by 1 bar: $p_t = s_{t-1}$.

    Returns
    -------
    results_df : pd.DataFrame
        DataFrame with strategy returns, cumulative equity, and trades.
    metrics : dict[str, float]
        Summary performance statistics.
    """
    df = pd.DataFrame(index=returns.index)
    df["market_return"] = returns

    if lag_positions:
        df["position"] = positions.shift(1).fillna(0.0)
    else:
        df["position"] = positions

    # Gross strategy return = active position * realized market return
    df["strategy_gross"] = df["position"] * df["market_return"]

    # Turnover / switches (measured on actual position vector)
    df["trades"] = (
        df["position"].diff().abs().fillna(df["position"].abs())
    )
    df["cost"] = df["trades"] * tc
    df["strategy_net"] = df["strategy_gross"] - df["cost"]

    # Compounded cumulative equity curves
    df["creturns_market"] = np.exp(df["market_return"].cumsum())
    df["creturns_gross"] = np.exp(df["strategy_gross"].cumsum())
    df["creturns_net"] = np.exp(df["strategy_net"].cumsum())

    # Running maximum and drawdown series
    cum_net = df["creturns_net"]
    running_max = cum_net.cummax()
    drawdown = (cum_net - running_max) / running_max
    df["drawdown"] = drawdown

    # Annualized metrics
    total_days = len(df)
    years = max(total_days / periods_per_year, 0.01)

    ann_ret_market = np.exp(df["market_return"].sum() / years) - 1.0
    ann_ret_net = np.exp(df["strategy_net"].sum() / years) - 1.0

    vol_market = df["market_return"].std() * np.sqrt(periods_per_year)
    vol_net = df["strategy_net"].std() * np.sqrt(periods_per_year)

    sharpe_market = (ann_ret_market / vol_market) if vol_market > 0 else 0.0
    sharpe_net = (ann_ret_net / vol_net) if vol_net > 0 else 0.0

    max_dd = drawdown.min()
    total_switches = int(df["trades"].sum())
    non_zero_bars = (df["strategy_gross"] != 0).sum()
    if non_zero_bars > 0:
        bar_win_rate = (df["strategy_gross"] > 0).sum() / non_zero_bars
    else:
        bar_win_rate = 0.0

    metrics = {
        "Annualized Market Return": ann_ret_market,
        "Annualized Strategy Net Return": ann_ret_net,
        "Annualized Volatility": vol_net,
        "Sharpe Ratio (Market)": sharpe_market,
        "Sharpe Ratio (Strategy Net)": sharpe_net,
        "Maximum Drawdown": max_dd,
        "Position Switches (Turnover Count)": total_switches,
        "Bar Win Rate (Gross)": bar_win_rate,
    }

    return df, metrics
