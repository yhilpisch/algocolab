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


class ProductionDNN(nn.Module):
    def __init__(
        self,
        input_dim: int = 7,
        hidden_units: list[int] = [64, 32]
    ):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_units:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ZMQTradingClient:
    """Consumes real-time ZeroMQ tick stream, computes rolling features,
    executes PyTorch model inference, enforces risk limits, and persists state.
    """

    def __init__(
        self,
        connect_addr: str = "tcp://127.0.0.1:5555",
        model: nn.Module | None = None,
        model_path: str = "best_trading_dnn.pt",
        initial_capital: float = 100_000.0,
        max_drawdown_limit: float = 0.10,
        tc_rate: float = 0.0005,
        db_path: str = "webinar_live_trading.db"
    ):
        self.connect_addr = connect_addr
        self.db_path = db_path
        self.cash = initial_capital
        self.nav = initial_capital
        self.peak_nav = initial_capital
        self.position = 0
        self.units_per_trade = 100
        self.max_dd_limit = max_drawdown_limit
        self.tc_rate = tc_rate
        self.halted = False
        self.prices: list[float] = []
        self.timestamps: list[str] = []

        if model is not None:
            self.model = model
        else:
            self.model = ProductionDNN(input_dim=7, hidden_units=[64, 32])
            if Path(model_path).exists():
                weights = torch.load(model_path, map_location="cpu")
                self.model.load_state_dict(weights)
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

        if len(self.prices) < 25 or self.halted:
            return

        p_arr = np.array(self.prices[-25:])
        rets = np.diff(np.log(p_arr))
        lag_features = rets[-5:][::-1]
        vol_20 = np.std(rets[-20:])
        mom_10 = np.mean(rets[-10:])
        feat_vector = np.concatenate([lag_features, [vol_20, mom_10]])

        x_t = torch.tensor(feat_vector, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            prob = torch.sigmoid(self.model(x_t)).item()

        target_pos = 1 if prob > 0.52 else (-1 if prob < 0.48 else 0)

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
                        f"Price: {p_val:.2f} | "
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
