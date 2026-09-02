"""
Vectorized backtesting and performance analytics for quantitative strategies.

(c) Dr. Yves J. Hilpisch
The Python Quants GmbH | https://tpq.io
https://hilpisch.com
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def run_vectorized_backtest(
    returns: pd.Series,
    positions: pd.Series,
    tc: float = 0.0005,  # 5 bps transaction cost per turnover
    periods_per_year: int = 252
) -> tuple[pd.DataFrame, dict[str, float]]:
    r"""Execute a vectorized backtest with transaction costs.

    Parameters
    ----------
    returns : pd.Series
        Asset log-returns $r_t = \ln(P_t / P_{t-1})$.
    positions : pd.Series
        Target positions $p_t \in \{-1, 0, 1\}$ (lagged to avoid lookahead).
    tc : float
        Proportional transaction cost per unit turnover.
    periods_per_year : int
        Annualization factor (252 for daily trading).

    Returns
    -------
    results_df : pd.DataFrame
        DataFrame with strategy returns, cumulative equity, and trades.
    metrics : dict[str, float]
        Summary performance statistics.
    """
    df = pd.DataFrame(index=returns.index)
    df["market_return"] = returns
    df["position"] = positions
    df["strategy_gross"] = df["position"] * df["market_return"]

    df["trades"] = (
        df["position"].diff().abs().fillna(df["position"].abs())
    )
    df["cost"] = df["trades"] * tc
    df["strategy_net"] = df["strategy_gross"] - df["cost"]

    df["creturns_market"] = np.exp(df["market_return"].cumsum())
    df["creturns_gross"] = np.exp(df["strategy_gross"].cumsum())
    df["creturns_net"] = np.exp(df["strategy_net"].cumsum())

    cum_net = df["creturns_net"]
    running_max = cum_net.cummax()
    drawdown = (cum_net - running_max) / running_max
    df["drawdown"] = drawdown

    total_days = len(df)
    years = max(total_days / periods_per_year, 0.01)

    ann_ret_market = np.exp(df["market_return"].sum() / years) - 1.0
    ann_ret_net = np.exp(df["strategy_net"].sum() / years) - 1.0

    vol_market = df["market_return"].std() * np.sqrt(periods_per_year)
    vol_net = df["strategy_net"].std() * np.sqrt(periods_per_year)

    sharpe_market = (ann_ret_market / vol_market) if vol_market > 0 else 0.0
    sharpe_net = (ann_ret_net / vol_net) if vol_net > 0 else 0.0

    max_dd = drawdown.min()
    total_trades = int(df["trades"].sum())
    non_zero_trades = (df["strategy_gross"] != 0).sum()
    if non_zero_trades > 0:
        hit_ratio = (df["strategy_gross"] > 0).sum() / non_zero_trades
    else:
        hit_ratio = 0.0

    metrics = {
        "Annualized Market Return": ann_ret_market,
        "Annualized Strategy Net Return": ann_ret_net,
        "Annualized Volatility": vol_net,
        "Sharpe Ratio (Market)": sharpe_market,
        "Sharpe Ratio (Strategy Net)": sharpe_net,
        "Maximum Drawdown": max_dd,
        "Total Trades": total_trades,
        "Hit Ratio": hit_ratio,
    }

    return df, metrics
