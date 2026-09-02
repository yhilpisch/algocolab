"""
Streaming execution engine, SQLite persistence, and risk management.

(c) Dr. Yves J. Hilpisch
The Python Quants GmbH | https://tpq.io
https://hilpisch.com
"""

from __future__ import annotations

import sqlite3
import time
import logging
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.data import build_feature_vector


@dataclass
class MarketTick:
    timestamp: str
    symbol: str
    price: float


class SQLitePersistence:
    """Manages transactional state and audit logging in SQLite."""

    def __init__(self, db_path: str = "trading_system.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._create_tables()

    def _create_tables(self) -> None:
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS ticks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    price REAL NOT NULL
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    probability REAL NOT NULL,
                    signal INTEGER NOT NULL
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    units INTEGER NOT NULL,
                    price REAL NOT NULL,
                    cost REAL NOT NULL
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_state (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    cash REAL NOT NULL,
                    nav REAL NOT NULL,
                    drawdown REAL NOT NULL
                )
            """)

    def save_tick(self, tick: MarketTick) -> None:
        sql = "INSERT INTO ticks (timestamp, symbol, price) VALUES (?, ?, ?)"
        with self.conn:
            self.conn.execute(sql, (tick.timestamp, tick.symbol, tick.price))

    def save_signal(
        self,
        timestamp: str,
        symbol: str,
        proba: float,
        signal: int
    ) -> None:
        sql = (
            "INSERT INTO signals (timestamp, symbol, probability, signal) "
            "VALUES (?, ?, ?, ?)"
        )
        with self.conn:
            self.conn.execute(sql, (timestamp, symbol, proba, signal))

    def save_order(
        self,
        timestamp: str,
        symbol: str,
        side: str,
        units: int,
        price: float,
        cost: float
    ) -> None:
        sql = (
            "INSERT INTO orders (timestamp, symbol, side, units, price, cost) "
            "VALUES (?, ?, ?, ?, ?, ?)"
        )
        with self.conn:
            self.conn.execute(
                sql, (timestamp, symbol, side, units, price, cost)
            )

    def save_state(
        self,
        timestamp: str,
        position: int,
        cash: float,
        nav: float,
        drawdown: float
    ) -> None:
        sql = (
            "INSERT INTO portfolio_state "
            "(timestamp, position, cash, nav, drawdown) "
            "VALUES (?, ?, ?, ?, ?)"
        )
        with self.conn:
            self.conn.execute(
                sql, (timestamp, position, cash, nav, drawdown)
            )

    def close(self) -> None:
        self.conn.close()


class RiskGuardrail:
    """Monitors operational risk limits (maximum drawdown)."""

    def __init__(
        self,
        max_drawdown_limit: float = 0.15,
        max_position_units: int = 1000,
        initial_capital: float = 100_000.0
    ):
        self.max_drawdown_limit = max_drawdown_limit
        self.max_position_units = max_position_units
        self.is_halted = False
        self.peak_nav = initial_capital

    def check(self, current_nav: float) -> bool:
        if current_nav > self.peak_nav:
            self.peak_nav = current_nav
        if self.peak_nav > 0:
            current_dd = (self.peak_nav - current_nav) / self.peak_nav
            if current_dd >= self.max_drawdown_limit:
                self.is_halted = True
                logging.error(
                    f"CIRCUIT BREAKER TRIGGERED: Current Drawdown "
                    f"{current_dd:.2%} >= {self.max_drawdown_limit:.2%}"
                )
                return False
        return not self.is_halted


class LiveTradingSimulation:
    """End-to-end trading system simulation replaying market observations."""

    def __init__(
        self,
        model: nn.Module,
        feature_dim: int = 5,
        scaler_mean: np.ndarray | None = None,
        scaler_scale: np.ndarray | None = None,
        initial_cash: float = 100_000.0,
        upper_threshold: float = 0.55,
        lower_threshold: float = 0.45,
        tc_rate: float = 0.0005,
        db_path: str = "trading_system.db"
    ):
        self.model = model.eval()
        self.feature_dim = feature_dim
        self.scaler_mean = scaler_mean
        self.scaler_scale = scaler_scale
        self.cash = initial_cash
        self.nav = initial_cash
        self.position = 0
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.tc_rate = tc_rate
        self.db = SQLitePersistence(db_path)
        self.risk = RiskGuardrail(initial_capital=initial_cash)
        self.price_history: list[float] = []

    def on_tick(self, tick: MarketTick) -> None:
        self.db.save_tick(tick)
        self.price_history.append(tick.price)

        # Mark NAV to market before check
        self.nav = self.cash + (self.position * 100 * tick.price)

        if len(self.price_history) < 22 or self.risk.is_halted:
            return

        if not self.risk.check(self.nav):
            return

        # Canonical unified feature extraction (C2)
        feat_vector = build_feature_vector(
            self.price_history[-25:], lags=self.feature_dim,
            vol_window=20, mom_window=10
        )

        # Standardize features (C3)
        if self.scaler_mean is not None and self.scaler_scale is not None:
            feat_vector = (
                feat_vector - self.scaler_mean
            ) / self.scaler_scale

        feat_tensor = torch.tensor(
            feat_vector, dtype=torch.float32
        ).unsqueeze(0)
        with torch.no_grad():
            prob = torch.sigmoid(self.model(feat_tensor)).item()

        if prob > self.upper_threshold:
            target_pos = 1
        elif prob < self.lower_threshold:
            target_pos = -1
        else:
            target_pos = 0

        self.db.save_signal(tick.timestamp, tick.symbol, prob, target_pos)

        pos_delta = target_pos - self.position
        if pos_delta != 0:
            units = abs(pos_delta) * 100
            side = "BUY" if pos_delta > 0 else "SELL"
            trade_cost = units * tick.price * self.tc_rate
            self.cash -= (pos_delta * 100 * tick.price + trade_cost)
            self.position = target_pos
            self.db.save_order(
                tick.timestamp, tick.symbol, side,
                units, tick.price, trade_cost
            )

        self.nav = self.cash + (self.position * 100 * tick.price)
        peak = self.risk.peak_nav
        drawdown = ((peak - self.nav) / peak) if peak > 0 else 0.0
        self.db.save_state(
            tick.timestamp, self.position, self.cash, self.nav, drawdown
        )
