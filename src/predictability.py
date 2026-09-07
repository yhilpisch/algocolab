"""
Predictability diagnostics for the EMH null-hypothesis experiment.

(c) Dr. Yves J. Hilpisch
The Python Quants GmbH | https://tpq.io
https://hilpisch.com
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox


def simulate_return_controls(
    observations: int,
    mean: float,
    volatility: float,
    persistence: float = 0.30,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate independent and AR(1) returns with comparable moments."""
    if observations < 3:
        raise ValueError("At least three observations are required.")
    if volatility <= 0.0:
        raise ValueError("Volatility must be positive.")
    if not -1.0 < persistence < 1.0:
        raise ValueError("AR(1) persistence must lie in (-1, 1).")

    rng = np.random.default_rng(seed)
    independent = rng.normal(mean, volatility, observations)
    innovation_scale = volatility * np.sqrt(1.0 - persistence**2)
    innovations = rng.normal(0.0, innovation_scale, observations)
    predictable = np.empty(observations, dtype=float)
    predictable[0] = mean + innovations[0]

    for index in range(1, observations):
        predictable[index] = (
            mean
            + persistence * (predictable[index - 1] - mean)
            + innovations[index]
        )

    return pd.DataFrame(
        {
            "independent": independent,
            "predictable_ar1": predictable,
        }
    )


def distribution_summary(returns: pd.Series) -> dict[str, float]:
    """Summarize moments relevant to the Gaussian-return null model."""
    clean = pd.Series(returns, dtype=float).dropna()
    if len(clean) < 3:
        raise ValueError("At least three finite returns are required.")
    return {
        "observations": float(len(clean)),
        "mean": float(clean.mean()),
        "volatility": float(clean.std(ddof=1)),
        "skewness": float(stats.skew(clean, bias=False)),
        "excess_kurtosis": float(stats.kurtosis(clean, bias=False)),
    }


def autocorrelation_table(
    returns: pd.Series,
    lags: int = 10,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Calculate sample autocorrelations and approximate null bounds."""
    clean = pd.Series(returns, dtype=float).dropna()
    if lags < 1 or lags >= len(clean):
        raise ValueError("Lags must be positive and below sample size.")
    critical_value = float(stats.norm.ppf(1.0 - alpha / 2.0))
    bound = critical_value / np.sqrt(len(clean))
    values = [clean.autocorr(lag=lag) for lag in range(1, lags + 1)]
    table = pd.DataFrame(
        {
            "lag": np.arange(1, lags + 1),
            "autocorrelation": values,
            "lower_bound": -bound,
            "upper_bound": bound,
        }
    )
    table["outside_bound"] = (
        table["autocorrelation"].abs() > bound
    )
    return table


def ljung_box_table(
    returns: pd.Series,
    lags: Iterable[int] = (5, 10, 20),
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Test the joint null of zero autocorrelation through each lag."""
    clean = pd.Series(returns, dtype=float).dropna()
    selected_lags = sorted({int(lag) for lag in lags})
    if not selected_lags or selected_lags[0] < 1:
        raise ValueError("At least one positive lag is required.")
    if selected_lags[-1] >= len(clean):
        raise ValueError("Every lag must be below the sample size.")

    table = acorr_ljungbox(
        clean,
        lags=selected_lags,
        return_df=True,
    ).reset_index(names="lag")
    table = table.rename(
        columns={"lb_stat": "statistic", "lb_pvalue": "p_value"}
    )
    table["alpha"] = alpha
    table["reject_no_autocorrelation"] = table["p_value"] < alpha
    return table


def serial_dependence_suite(
    series: Mapping[str, pd.Series],
    lags: Iterable[int] = (5, 10, 20),
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run the Ljung-Box diagnostic over empirical and control series."""
    frames = []
    for name, values in series.items():
        result = ljung_box_table(values, lags=lags, alpha=alpha)
        result.insert(0, "series", name)
        frames.append(result)
    if not frames:
        raise ValueError("At least one return series is required.")
    return pd.concat(frames, ignore_index=True)

