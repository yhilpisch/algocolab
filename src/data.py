"""
Data loading and feature engineering for algorithmic trading.

(c) Dr. Yves J. Hilpisch
The Python Quants GmbH | https://tpq.io
https://hilpisch.com
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

DEFAULT_DATA_URL = "https://hilpisch.com/eod_data.csv"


def load_eod_data(
    source: str | Path = DEFAULT_DATA_URL,
    symbol: str = "SPY",
    date_col: str = "Date"
) -> pd.Series:
    """Load end-of-day price series for a single symbol.

    Parameters
    ----------
    source : str or Path
        Local file path or remote URL.
    symbol : str
        Column name for asset (e.g. 'SPY', 'EURUSD', 'BTC-USD', 'AAPL').
    date_col : str
        Name of the date index column.

    Returns
    -------
    pd.Series
        Cleaned time series of prices indexed by DatetimeIndex.
    """
    path = Path(str(source))
    if path.is_file():
        df = pd.read_csv(path, parse_dates=[date_col])
    else:
        df = pd.read_csv(source, parse_dates=[date_col])
    df = df.set_index(date_col).sort_index()
    if symbol not in df.columns:
        cols_msg = list(df.columns)
        raise ValueError(f"Symbol '{symbol}' not found in columns: {cols_msg}")
    prices = df[symbol].astype(float).dropna()
    return prices


def create_lagged_features(
    prices: pd.Series,
    lags: int = 5,
    include_volatility: bool = True,
    vol_window: int = 20,
    include_momentum: bool = True,
    mom_window: int = 10
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Create lagged returns, rolling volatility, and momentum features.

    Parameters
    ----------
    prices : pd.Series
        Price time series.
    lags : int
        Number of return lags to generate.
    include_volatility : bool
        Whether to include rolling return standard deviation.
    vol_window : int
        Rolling window size for volatility.
    include_momentum : bool
        Whether to include rolling cumulative return momentum.
    mom_window : int
        Rolling window size for momentum.

    Returns
    -------
    features : pd.DataFrame
        Matrix of feature vectors $X_t$.
    returns : pd.Series
        Actual next-period log-return $r_{t+1}$.
    direction : pd.Series
        Binary classification target: 1 if $r_{t+1} > 0$, 0 otherwise.
    """
    df = pd.DataFrame(index=prices.index)
    df["price"] = prices
    df["return"] = np.log(prices / prices.shift(1))

    feature_cols = []
    for lag in range(1, lags + 1):
        col_name = f"lag_{lag}"
        df[col_name] = df["return"].shift(lag)
        feature_cols.append(col_name)

    if include_volatility:
        df["rolling_vol"] = df["return"].shift(1).rolling(vol_window).std()
        feature_cols.append("rolling_vol")

    if include_momentum:
        df["rolling_mom"] = df["return"].shift(1).rolling(mom_window).mean()
        feature_cols.append("rolling_mom")

    df["target_return"] = df["return"]
    df["target_direction"] = (df["target_return"] > 0).astype(int)

    df = df.dropna()

    X = df[feature_cols].copy()
    y_return = df["target_return"].copy()
    y_direction = df["target_direction"].copy()

    return X, y_return, y_direction


def train_val_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.Series, pd.Series, pd.Series
]:
    """Split time-series data chronologically without lookahead bias.

    Parameters
    ----------
    X : pd.DataFrame
        Features.
    y : pd.Series
        Targets.
    train_ratio : float
        Proportion for training set.
    val_ratio : float
        Proportion for validation set.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

    return X_train, X_val, X_test, y_train, y_val, y_test
