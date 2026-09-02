"""
Automated unit and integration test suite for the trading system.

(c) Dr. Yves J. Hilpisch
The Python Quants GmbH | https://tpq.io
https://hilpisch.com
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from src.data import create_lagged_features, build_feature_vector
from src.backtest import run_vectorized_backtest
from src.engine import RiskGuardrail, LiveTradingSimulation, MarketTick
from src.models import TradingDNN


def test_feature_parity():
    """Verify feature parity: batch DataFrame vs rolling window (C2)."""
    np.random.seed(42)
    prices = pd.Series(
        100.0 * np.exp(np.cumsum(np.random.normal(0.0005, 0.01, 100))),
        index=pd.date_range("2025-01-01", periods=100, freq="D")
    )
    df_feat, _, _ = create_lagged_features(
        prices, lags=5, vol_window=20, mom_window=10
    )

    # In batch mode, row t uses returns up to t-1 (prices up to t-1)
    # So to generate features for row t, window contains prices up to t-1
    window_up_to_t_minus_1 = prices.iloc[:-1].iloc[-25:].values
    vec_feat = build_feature_vector(
        window_up_to_t_minus_1, lags=5, vol_window=20, mom_window=10
    )

    df_last_row = df_feat.iloc[-1].values
    np.testing.assert_allclose(
        df_last_row, vec_feat, rtol=1e-5,
        err_msg="Batch and streaming features must match identically."
    )
    print("✓ Test Feature Parity Passed.")


def test_circuit_breaker_trips():
    """Verify circuit breaker trips when drawdown exceeds risk limit (H1)."""
    risk = RiskGuardrail(max_drawdown_limit=0.10, initial_capital=100_000.0)

    # NAV rises to 120k (peak = 120k)
    assert risk.check(120_000.0) is True
    assert risk.is_halted is False

    # NAV drops to 110k (drawdown = (120-110)/120 = 8.33% < 10%)
    assert risk.check(110_000.0) is True
    assert risk.is_halted is False

    # NAV drops to 105k (drawdown = (120-105)/120 = 12.5% >= 10%)
    tripped = risk.check(105_000.0)
    assert tripped is False
    assert risk.is_halted is True

    # Subsequent checks must remain halted
    assert risk.check(115_000.0) is False
    print("✓ Test Circuit Breaker Tripping Passed.")


def test_backtest_lag_and_metrics():
    """Verify backtest lag enforcement and metric calculations (C1, M1, M2)."""
    rets = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
    signals = pd.Series([1, 1, -1, -1, 1])

    # Unlagged test
    df_unlagged, m_unlagged = run_vectorized_backtest(
        rets, signals, tc=0.0005, lag_positions=False
    )
    assert "Position Switches (Turnover Count)" in m_unlagged
    assert "Bar Win Rate (Gross)" in m_unlagged
    assert df_unlagged["position"].iloc[0] == 1

    # Lagged test (positions shifted by 1)
    df_lagged, m_lagged = run_vectorized_backtest(
        rets, signals, tc=0.0005, lag_positions=True
    )
    assert df_lagged["position"].iloc[0] == 0.0
    assert df_lagged["position"].iloc[1] == 1.0
    assert df_lagged["position"].iloc[3] == -1.0
    print("✓ Test Backtest Timing & Metrics Passed.")


def test_live_simulation_with_scaler():
    """Verify live simulation runs with scaler standardization (C3)."""
    model = TradingDNN(input_dim=7, hidden_units=[16, 8], dropout_rate=0.0)
    scaler_mean = np.zeros(7, dtype=np.float32)
    scaler_scale = np.ones(7, dtype=np.float32)

    sim = LiveTradingSimulation(
        model=model,
        feature_dim=5,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        initial_cash=100_000.0,
        upper_threshold=0.51,
        lower_threshold=0.49
    )

    # Feed 30 synthetic ticks
    price = 100.0
    for i in range(30):
        price += np.random.normal(0, 0.2)
        sim.on_tick(MarketTick(f"2025-01-01T10:00:{i:02d}", "SPY", price))

    assert len(sim.price_history) == 30
    assert sim.nav > 0
    print("✓ Test Live Simulation Scaler Standardization Passed.")


if __name__ == "__main__":
    print("Running system test suite...")
    test_feature_parity()
    test_circuit_breaker_trips()
    test_backtest_lag_and_metrics()
    test_live_simulation_with_scaler()
    print("\nAll 4 System Integration Tests Passed Successfully!")
