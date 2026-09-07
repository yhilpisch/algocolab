"""
Granger-style tests of incremental factor predictability.

(c) Dr. Yves J. Hilpisch
The Python Quants GmbH | https://tpq.io
https://hilpisch.com
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
import statsmodels.api as sm


def _lagged_design(
    target: pd.Series,
    factor: pd.Series,
    target_lags: int,
    factor_lags: int,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Build aligned restricted and full predictive regressions."""
    if target_lags < 1 or factor_lags < 1:
        raise ValueError("Lag counts must be positive.")

    target_name = target.name or "target"
    factor_name = factor.name or "factor"
    frame = pd.concat(
        [target.rename("target"), factor.rename("factor")],
        axis=1,
        join="inner",
    ).dropna()

    target_columns = []
    factor_columns = []
    for lag in range(1, target_lags + 1):
        name = f"{target_name}_lag_{lag}"
        frame[name] = frame["target"].shift(lag)
        target_columns.append(name)
    for lag in range(1, factor_lags + 1):
        name = f"{factor_name}_lag_{lag}"
        frame[name] = frame["factor"].shift(lag)
        factor_columns.append(name)

    frame = frame.dropna()
    if len(frame) <= len(target_columns) + len(factor_columns) + 1:
        raise ValueError("The aligned sample is too short for the models.")

    response = frame["target"]
    restricted = sm.add_constant(frame[target_columns], has_constant="add")
    full = sm.add_constant(
        frame[target_columns + factor_columns],
        has_constant="add",
    )
    return response, restricted, full


def granger_factor_test(
    target: pd.Series,
    factor: pd.Series,
    target_lags: int = 5,
    factor_lags: int = 5,
    alpha: float = 0.05,
) -> dict[str, float | int | bool | str]:
    """Test whether factor lags add predictive content for the target."""
    response, restricted_x, full_x = _lagged_design(
        target,
        factor,
        target_lags,
        factor_lags,
    )
    restricted_model = sm.OLS(response, restricted_x).fit()
    full_model = sm.OLS(response, full_x).fit()
    statistic, p_value, restrictions = full_model.compare_f_test(
        restricted_model
    )
    factor_name = factor.name or "factor"
    return {
        "factor": factor_name,
        "observations": int(full_model.nobs),
        "target_lags": target_lags,
        "factor_lags": factor_lags,
        "f_statistic": float(statistic),
        "p_value": float(p_value),
        "restrictions": int(restrictions),
        "alpha": alpha,
        "reject_no_incremental_content": bool(p_value < alpha),
        "restricted_adj_r2": float(restricted_model.rsquared_adj),
        "full_adj_r2": float(full_model.rsquared_adj),
        "incremental_adj_r2": float(
            full_model.rsquared_adj - restricted_model.rsquared_adj
        ),
    }


def factor_test_suite(
    target: pd.Series,
    factors: Mapping[str, pd.Series],
    target_lags: int = 5,
    factor_lags: int = 5,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Apply the restricted/full-model test to candidate factors."""
    rows = []
    for name, factor in factors.items():
        named_factor = pd.Series(factor, copy=False).rename(name)
        rows.append(
            granger_factor_test(
                target,
                named_factor,
                target_lags=target_lags,
                factor_lags=factor_lags,
                alpha=alpha,
            )
        )
    if not rows:
        raise ValueError("At least one candidate factor is required.")
    return pd.DataFrame(rows)
