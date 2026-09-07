"""
ZeroMQ SUB trading client performing real-time inference & order management.

(c) Dr. Yves J. Hilpisch
The Python Quants GmbH | https://tpq.io
https://hilpisch.com
"""

from __future__ import annotations

import sqlite3
import numpy as np
import zmq
import torch
import torch.nn as nn
from pathlib import Path

from src.config import FX_COST_ONE_WAY
from src.data import build_feature_vector
from src.models import load_model_checkpoint


class ZMQTradingClient:
    """Consumes real-time ZeroMQ tick stream, computes canonical features,
    applies training standardization, performs PyTorch inference,
    enforces risk limits, and persists state to SQLite.
    """

    def __init__(
        self,
        connect_addr: str = "tcp://127.0.0.1:5555",
        model: nn.Module | None = None,
        model_path: str = "best_trading_dnn.pt",
        scaler_mean: np.ndarray | None = None,
        scaler_scale: np.ndarray | None = None,
        threshold: float | None = None,
        initial_capital: float = 100_000.0,
        max_drawdown_limit: float = 0.10,
        tc_rate: float = FX_COST_ONE_WAY,
        db_path: str = "webinar_live_trading.db"
    ):
        self.connect_addr = connect_addr
        self.db_path = db_path
        self.cash = initial_capital
        self.nav = initial_capital
        self.peak_nav = initial_capital
        self.position = 0
        self.units_per_trade = 10_000
        self.max_dd_limit = max_drawdown_limit
        self.tc_rate = tc_rate
        self.halted = False
        self.prices: list[float] = []
        self.timestamps: list[str] = []

        if model is not None:
            self.model = model
            self.scaler_mean = scaler_mean
            self.scaler_scale = scaler_scale
            self.threshold = threshold
            self.feature_lags = 5
        else:
            checkpoint_path = Path(model_path)
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"Required model checkpoint not found: {checkpoint_path}"
                )
            self.model, payload = load_model_checkpoint(checkpoint_path)
            self.scaler_mean = payload["scaler_mean"].numpy()
            self.scaler_scale = payload["scaler_scale"].numpy()
            self.threshold = float(payload["threshold"])
            self.feature_lags = sum(
                name.startswith("lag_")
                for name in payload["feature_names"]
            )
        if self.scaler_mean is None or self.scaler_scale is None:
            raise ValueError("A fitted training scaler is required.")
        if self.threshold is None or not 0.5 <= self.threshold < 1.0:
            raise ValueError("A valid symmetric threshold is required.")
        self.model.eval()

        self._init_db()

    def _init_db(self) -> None:
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS ticks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT, symbol TEXT, price REAL
                )""")
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT, symbol TEXT, prob_up REAL, signal INTEGER
                )""")
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT, symbol TEXT, side TEXT,
                    units INTEGER, price REAL, cost REAL
                )""")
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_state (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT, position INTEGER, cash REAL,
                    nav REAL, drawdown REAL
                )""")

    def process_tick(self, timestamp: str, symbol: str, price: float) -> None:
        self.prices.append(price)
        self.timestamps.append(timestamp)

        sql_tick = (
            "INSERT INTO ticks (timestamp, symbol, price) VALUES (?, ?, ?)"
        )
        with self.conn:
            self.conn.execute(sql_tick, (timestamp, symbol, price))

        # Require at least 22 prices (20-vol window + 2)
        if len(self.prices) < 22 or self.halted:
            return

        # Canonical unified feature extraction (C2)
        feat_vector = build_feature_vector(
            self.prices[-25:],
            lags=self.feature_lags,
            vol_window=20,
            mom_window=10,
        )

        # Standardize features using training distribution (C3)
        if self.scaler_mean is not None and self.scaler_scale is not None:
            feat_vector = (
                feat_vector - self.scaler_mean
            ) / self.scaler_scale

        x_t = torch.tensor(feat_vector, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            prob = torch.sigmoid(self.model(x_t)).item()

        lower_threshold = 1.0 - self.threshold
        target_pos = (
            1
            if prob > self.threshold
            else (-1 if prob < lower_threshold else 0)
        )

        sql_sig = (
            "INSERT INTO signals (timestamp, symbol, prob_up, signal) "
            "VALUES (?, ?, ?, ?)"
        )
        with self.conn:
            self.conn.execute(
                sql_sig, (timestamp, symbol, prob, target_pos)
            )

        delta = target_pos - self.position
        if delta != 0:
            units = abs(delta) * self.units_per_trade
            side = "BUY" if delta > 0 else "SELL"
            cost = units * price * self.tc_rate
            self.cash -= (delta * self.units_per_trade * price + cost)
            self.position = target_pos
            sql_ord = (
                "INSERT INTO orders "
                "(timestamp, symbol, side, units, price, cost) "
                "VALUES (?, ?, ?, ?, ?, ?)"
            )
            with self.conn:
                self.conn.execute(
                    sql_ord, (timestamp, symbol, side, units, price, cost)
                )

        # Mark to market & Drawdown check (H1)
        self.nav = self.cash + (self.position * self.units_per_trade * price)
        if self.nav > self.peak_nav:
            self.peak_nav = self.nav
        peak = self.peak_nav
        dd = ((peak - self.nav) / peak) if peak > 0 else 0.0

        if dd >= self.max_dd_limit:
            self.halted = True
            print(f"[{timestamp}] CIRCUIT BREAKER TRIPPED! Drawdown: {dd:.2%}")

        sql_state = (
            "INSERT INTO portfolio_state "
            "(timestamp, position, cash, nav, drawdown) "
            "VALUES (?, ?, ?, ?, ?)"
        )
        with self.conn:
            self.conn.execute(
                sql_state, (timestamp, self.position, self.cash, self.nav, dd)
            )

    def run(
        self,
        max_ticks: int | None = 100,
        timeout_ms: int = 500,
        max_idle_timeouts: int = 3
    ) -> None:
        ctx = zmq.Context()
        socket = ctx.socket(zmq.SUB)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        socket.connect(self.connect_addr)
        socket.setsockopt_string(zmq.SUBSCRIBE, "")

        print(
            f"[ZMQ Trading Engine] Connected to {self.connect_addr}. "
            f"Consuming live stream..."
        )
        count = 0
        idle_timeouts = 0
        try:
            while max_ticks is None or count < max_ticks:
                try:
                    tick = socket.recv_json()
                    idle_timeouts = 0
                except zmq.Again:
                    idle_timeouts += 1
                    if idle_timeouts >= max_idle_timeouts:
                        print(
                            "[ZMQ Trading Engine] Stream ended / "
                            "idle timeout reached."
                        )
                        break
                    continue
                except (zmq.ContextTerminated, zmq.ZMQError):
                    break
                self.process_tick(
                    tick["time"], tick["symbol"], float(tick["price"])
                )
                count += 1
                if count % 20 == 0:
                    t_str = tick['time'][:19]
                    p_val = tick['price']
                    print(
                        f"[{t_str}] Ticks: {count:03d} | "
                        f"Price: {p_val:.4f} | "
                        f"Pos: {self.position:+d} | "
                        f"NAV: ${self.nav:,.2f}"
                    )
        except (KeyboardInterrupt, SystemExit):
            pass
        finally:
            try:
                socket.close(0)
                self.conn.close()
                ctx.term()
            except Exception:
                pass
            print("[ZMQ Trading Engine] Stopped and closed.")


if __name__ == "__main__":
    client = ZMQTradingClient()
    client.run(max_ticks=100)
