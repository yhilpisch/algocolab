"""
Canonical Session 1 experiment and result persistence.

(c) Dr. Yves J. Hilpisch
The Python Quants GmbH | https://tpq.io
https://hilpisch.com
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.artifacts import RunBundle
from src.backtest import run_vectorized_backtest
from src.config import ExperimentConfig
from src.factor_tests import factor_test_suite
from src.predictability import (
    autocorrelation_table,
    distribution_summary,
    serial_dependence_suite,
    simulate_return_controls,
)


@dataclass
class SessionOneResults:
    """Tables produced by the canonical Session 1 experiment."""

    prices: pd.DataFrame
    returns: pd.DataFrame
    moments: pd.DataFrame
    autocorrelations: pd.DataFrame
    ljung_box: pd.DataFrame
    factor_tests: pd.DataFrame
    predictions: pd.DataFrame
    strategy_metrics: pd.DataFrame


def load_price_panel(
    data_path: str | Path,
    config: ExperimentConfig,
) -> pd.DataFrame:
    """Load the frozen price panel under the canonical sample dates."""
    prices = pd.read_csv(data_path, parse_dates=["Date"])
    prices = prices.set_index("Date").sort_index()
    prices = prices.loc[config.data_start : config.data_end]
    if config.symbol not in prices.columns:
        raise ValueError(f"Target symbol not found: {config.symbol}")
    if prices.empty:
        raise ValueError("No prices fall inside the configured sample.")
    return prices.astype(float)


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate aligned log returns for the full price panel."""
    returns = np.log(prices / prices.shift(1))
    return returns.replace([np.inf, -np.inf], np.nan).dropna()


def _moment_table(series: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for name, values in series.items():
        row = distribution_summary(values)
        row["series"] = name
        rows.append(row)
    columns = [
        "series",
        "observations",
        "mean",
        "volatility",
        "skewness",
        "excess_kurtosis",
    ]
    return pd.DataFrame(rows)[columns]


def _autocorrelation_suite(
    series: dict[str, pd.Series],
    lags: int,
    alpha: float,
) -> pd.DataFrame:
    frames = []
    for name, values in series.items():
        table = autocorrelation_table(values, lags=lags, alpha=alpha)
        table.insert(0, "series", name)
        frames.append(table)
    return pd.concat(frames, ignore_index=True)


def _lagged_target(
    target: pd.Series,
    lags: int,
) -> tuple[pd.DataFrame, list[str]]:
    frame = pd.DataFrame({"target_return": target})
    columns = []
    for lag in range(1, lags + 1):
        name = f"target_lag_{lag}"
        frame[name] = frame["target_return"].shift(lag)
        columns.append(name)
    return frame.dropna(), columns


def _strategy_experiment(
    target: pd.Series,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame, features = _lagged_target(target, config.target_lags)
    train_end = int(len(frame) * config.train_ratio)
    validation_end = int(
        len(frame) * (config.train_ratio + config.validation_ratio)
    )
    frame["sample"] = "test"
    frame.iloc[:train_end, frame.columns.get_loc("sample")] = "train"
    frame.iloc[
        train_end:validation_end,
        frame.columns.get_loc("sample"),
    ] = "validation"

    model = LinearRegression().fit(
        frame.iloc[:train_end][features],
        frame.iloc[:train_end]["target_return"],
    )
    frame["prediction"] = model.predict(frame[features])
    frame["position"] = np.sign(frame["prediction"])

    metric_rows = []
    output_frames = []
    for sample in ("train", "validation", "test"):
        selected = frame.loc[frame["sample"] == sample].copy()
        backtest, metrics = run_vectorized_backtest(
            selected["target_return"],
            selected["position"],
            tc=config.transaction_cost_one_way,
            periods_per_year=config.periods_per_year,
        )
        selected = selected.join(
            backtest[
                [
                    "strategy_gross",
                    "trades",
                    "cost",
                    "strategy_net",
                    "creturns_market",
                    "creturns_net",
                    "drawdown",
                ]
            ]
        )
        output_frames.append(selected)
        metric_rows.append(
            {
                "sample": sample,
                "start": selected.index.min().date().isoformat(),
                "end": selected.index.max().date().isoformat(),
                "observations": len(selected),
                "market_annual_return": metrics[
                    "Annualized Market Return"
                ],
                "strategy_net_annual_return": metrics[
                    "Annualized Strategy Net Return"
                ],
                "strategy_net_volatility": metrics[
                    "Annualized Volatility"
                ],
                "strategy_net_sharpe": metrics[
                    "Sharpe Ratio (Strategy Net)"
                ],
                "maximum_drawdown": metrics["Maximum Drawdown"],
                "turnover_units": metrics[
                    "Position Switches (Turnover Count)"
                ],
                "gross_bar_win_rate": metrics["Bar Win Rate (Gross)"],
            }
        )
    predictions = pd.concat(output_frames).sort_index()
    metrics = pd.DataFrame(metric_rows)
    return predictions, metrics


def run_session_one(
    data_path: str | Path,
    config: ExperimentConfig | None = None,
    factor_symbols: tuple[str, ...] = ("SPY", "GLD", "TLT"),
) -> SessionOneResults:
    """Run the complete Session 1 statistical skeleton."""
    active_config = config or ExperimentConfig()
    active_config.validate()
    prices = load_price_panel(data_path, active_config)
    returns = calculate_returns(prices)
    target = returns[active_config.symbol].rename(active_config.symbol)

    controls = simulate_return_controls(
        observations=len(target),
        mean=float(target.mean()),
        volatility=float(target.std(ddof=1)),
        persistence=0.30,
        seed=active_config.control_seed,
    )
    controls.index = target.index
    diagnostic_series = {
        "EURUSD empirical": target,
        "Independent control": controls["independent"],
        "AR(1) control": controls["predictable_ar1"],
    }
    moments = _moment_table(diagnostic_series)
    autocorrelations = _autocorrelation_suite(
        diagnostic_series,
        lags=10,
        alpha=active_config.significance_level,
    )
    ljung_box = serial_dependence_suite(
        diagnostic_series,
        lags=(5, 10, 20),
        alpha=active_config.significance_level,
    )

    missing = [name for name in factor_symbols if name not in returns]
    if missing:
        raise ValueError(f"Factor symbols not found: {missing}")
    factors = {name: returns[name] for name in factor_symbols}
    factor_tests = factor_test_suite(
        target,
        factors,
        target_lags=active_config.target_lags,
        factor_lags=active_config.factor_lags,
        alpha=active_config.significance_level,
    )
    predictions, strategy_metrics = _strategy_experiment(
        target,
        active_config,
    )
    return SessionOneResults(
        prices=prices,
        returns=returns,
        moments=moments,
        autocorrelations=autocorrelations,
        ljung_box=ljung_box,
        factor_tests=factor_tests,
        predictions=predictions,
        strategy_metrics=strategy_metrics,
    )


def persist_session_one(
    bundle: RunBundle,
    results: SessionOneResults,
) -> None:
    """Persist canonical Session 1 outputs and mark the session complete."""
    artifacts = {
        "session_1/moments.csv": results.moments,
        "session_1/autocorrelations.csv": results.autocorrelations,
        "session_1/ljung_box.csv": results.ljung_box,
        "session_1/factor_tests.csv": results.factor_tests,
        "session_1/predictions.csv": results.predictions.reset_index(),
        "session_1/strategy_metrics.csv": results.strategy_metrics,
    }
    for relative_path, frame in artifacts.items():
        bundle.write_frame(relative_path, frame)
    bundle.mark_session_complete(1, tuple(artifacts))
