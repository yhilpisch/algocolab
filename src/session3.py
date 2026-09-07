"""
Canonical Session 3 paper-trading replay and operational checks.

(c) Dr. Yves J. Hilpisch
The Python Quants GmbH | https://tpq.io
https://hilpisch.com
"""

from __future__ import annotations

import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch

from src.artifacts import RunBundle
from src.config import ExperimentConfig
from src.data import (
    build_feature_vector,
    create_lagged_features,
    load_eod_data,
)
from src.models import load_model_checkpoint


@dataclass
class SessionThreeResults:
    """Evidence produced by the Session 3 replay."""

    database_path: Path
    parity: pd.DataFrame
    telemetry: pd.DataFrame
    orders: pd.DataFrame
    failure_tests: pd.DataFrame
    reconciliation: dict[str, float | int | bool | str]


def validate_observation(
    previous_time: pd.Timestamp,
    current_time: pd.Timestamp,
    previous_price: float,
    current_price: float,
    max_gap_days: int = 5,
    max_jump: float = 0.15,
) -> None:
    """Reject stale or implausibly discontinuous market observations."""
    gap_days = (current_time - previous_time).days
    if gap_days > max_gap_days:
        raise ValueError(f"Stale observation gap: {gap_days} days")
    jump = abs(current_price / previous_price - 1.0)
    if jump > max_jump:
        raise ValueError(f"Price jump exceeds limit: {jump:.2%}")


class PaperReplay:
    """Replay target positions into an auditable SQLite ledger."""

    def __init__(
        self,
        database_path: str | Path,
        run_id: str,
        symbol: str,
        transaction_cost: float,
        max_drawdown: float = 0.10,
        initial_nav: float = 100_000.0,
    ) -> None:
        self.path = Path(database_path)
        if self.path.exists():
            raise FileExistsError(
                f"Replay database already exists: {self.path}"
            )
        self.run_id = run_id
        self.symbol = symbol
        self.transaction_cost = transaction_cost
        self.max_drawdown = max_drawdown
        self.nav = initial_nav
        self.peak_nav = initial_nav
        self.position = 0
        self.halted = False
        self.connection = sqlite3.connect(self.path)
        self._create_schema()

    def _create_schema(self) -> None:
        statements = (
            """CREATE TABLE ticks (
                run_id TEXT, timestamp TEXT, symbol TEXT, price REAL
            )""",
            """CREATE TABLE signals (
                run_id TEXT, timestamp TEXT, probability REAL,
                requested_position INTEGER
            )""",
            """CREATE TABLE orders (
                run_id TEXT, timestamp TEXT, from_position INTEGER,
                to_position INTEGER, turnover REAL, cost_fraction REAL,
                reason TEXT
            )""",
            """CREATE TABLE portfolio_state (
                run_id TEXT, timestamp TEXT, position INTEGER, nav REAL,
                drawdown REAL, halted INTEGER
            )""",
        )
        with self.connection:
            for statement in statements:
                self.connection.execute(statement)

    def _save_order(
        self,
        timestamp: pd.Timestamp,
        target: int,
        reason: str,
    ) -> float:
        turnover = abs(target - self.position)
        if turnover == 0:
            return 0.0
        cost_fraction = turnover * self.transaction_cost
        with self.connection:
            self.connection.execute(
                "INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    self.run_id,
                    timestamp.isoformat(),
                    self.position,
                    target,
                    turnover,
                    cost_fraction,
                    reason,
                ),
            )
        self.position = target
        return cost_fraction

    def process_bar(
        self,
        timestamp: pd.Timestamp,
        price: float,
        realized_return: float,
        probability: float,
        requested_position: int,
    ) -> None:
        """Process one close-to-close return with start-of-bar positioning."""
        with self.connection:
            self.connection.execute(
                "INSERT INTO ticks VALUES (?, ?, ?, ?)",
                (self.run_id, timestamp.isoformat(), self.symbol, price),
            )
            self.connection.execute(
                "INSERT INTO signals VALUES (?, ?, ?, ?)",
                (
                    self.run_id,
                    timestamp.isoformat(),
                    probability,
                    requested_position,
                ),
            )
        target = 0 if self.halted else requested_position
        entry_cost = self._save_order(timestamp, target, "signal")
        strategy_return = self.position * realized_return - entry_cost
        self.nav *= float(np.exp(strategy_return))
        self.peak_nav = max(self.peak_nav, self.nav)
        drawdown = self.nav / self.peak_nav - 1.0
        if drawdown <= -self.max_drawdown and not self.halted:
            exit_cost = self._save_order(timestamp, 0, "risk_flatten")
            self.nav *= float(np.exp(-exit_cost))
            self.halted = True
            drawdown = self.nav / self.peak_nav - 1.0
        with self.connection:
            self.connection.execute(
                "INSERT INTO portfolio_state VALUES (?, ?, ?, ?, ?, ?)",
                (
                    self.run_id,
                    timestamp.isoformat(),
                    self.position,
                    self.nav,
                    drawdown,
                    int(self.halted),
                ),
            )

    def frames(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Read telemetry and orders from the durable ledger."""
        telemetry = pd.read_sql_query(
            "SELECT * FROM portfolio_state ORDER BY timestamp",
            self.connection,
        )
        orders = pd.read_sql_query(
            "SELECT * FROM orders ORDER BY rowid",
            self.connection,
        )
        return telemetry, orders

    def close(self) -> None:
        """Close the SQLite connection."""
        self.connection.close()


def _model_inputs(
    data_path: str | Path,
    config: ExperimentConfig,
) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    prices = load_eod_data(data_path, symbol=config.symbol)
    prices = prices.loc[config.data_start : config.data_end]
    features, returns, _ = create_lagged_features(
        prices,
        lags=config.target_lags,
    )
    test_start = int(
        len(features) * (config.train_ratio + config.validation_ratio)
    )
    return prices, features.iloc[test_start:], returns.iloc[test_start:]


def _failure_tests(transaction_cost: float) -> pd.DataFrame:
    cases = []
    checks = (
        (
            "stale_observation",
            pd.Timestamp("2026-01-01"),
            pd.Timestamp("2026-01-10"),
            1.10,
            1.10,
        ),
        (
            "price_jump",
            pd.Timestamp("2026-01-01"),
            pd.Timestamp("2026-01-02"),
            1.10,
            1.30,
        ),
    )
    for name, previous_time, current_time, old_price, new_price in checks:
        rejected = False
        message = ""
        try:
            validate_observation(
                previous_time,
                current_time,
                old_price,
                new_price,
            )
        except ValueError as error:
            rejected = True
            message = str(error)
        cases.append(
            {"test": name, "passed": rejected, "evidence": message}
        )
    with TemporaryDirectory() as temporary:
        replay = PaperReplay(
            Path(temporary) / "risk.db",
            "failure-test",
            "EURUSD",
            transaction_cost,
            max_drawdown=0.01,
        )
        replay.process_bar(
            pd.Timestamp("2026-01-02"),
            1.10,
            -0.02,
            0.90,
            1,
        )
        telemetry, orders = replay.frames()
        replay.close()
    final_state = telemetry.iloc[-1]
    risk_orders = orders.query("reason == 'risk_flatten'")
    flattened = bool(
        final_state["halted"]
        and final_state["position"] == 0
        and len(risk_orders) == 1
    )
    cases.append(
        {
            "test": "drawdown_flatten",
            "passed": flattened,
            "evidence": (
                f"halted={bool(final_state['halted'])}, "
                f"final_position={int(final_state['position'])}"
            ),
        }
    )
    return pd.DataFrame(cases)


def run_session_three(
    bundle: RunBundle,
    data_path: str | Path,
    database_path: str | Path,
    max_drawdown: float = 0.10,
) -> SessionThreeResults:
    """Validate the model contract and execute the test-period replay."""
    bundle.validate(required_session=2)
    config = ExperimentConfig(**bundle.manifest["configuration"])
    model, payload = load_model_checkpoint(
        bundle.path / "session_2/model.pt"
    )
    if payload["run_id"] != bundle.run_id:
        raise ValueError("Checkpoint run ID does not match the bundle.")

    prices, features, returns = _model_inputs(data_path, config)
    if list(features.columns) != payload["feature_names"]:
        raise ValueError("Live feature order differs from training.")
    mean = payload["scaler_mean"].numpy()
    scale = payload["scaler_scale"].numpy()
    batch_values = (features.to_numpy() - mean) / scale
    batch_tensor = torch.tensor(batch_values, dtype=torch.float32)
    with torch.no_grad():
        batch_probability = (
            torch.sigmoid(model(batch_tensor)).numpy().ravel()
        )

    stream_features = []
    for timestamp in features.index:
        location = prices.index.get_loc(timestamp)
        history = prices.iloc[:location].to_numpy()
        stream_features.append(
            build_feature_vector(
                history,
                lags=config.target_lags,
                vol_window=20,
                mom_window=10,
            )
        )
    stream_values = (np.asarray(stream_features) - mean) / scale
    stream_tensor = torch.tensor(stream_values, dtype=torch.float32)
    with torch.no_grad():
        stream_probability = (
            torch.sigmoid(model(stream_tensor)).numpy().ravel()
        )
    threshold = float(payload["threshold"])
    positions = np.where(
        stream_probability > threshold,
        1,
        np.where(stream_probability < 1.0 - threshold, -1, 0),
    )
    saved = pd.read_csv(
        bundle.path / "session_2/predictions.csv",
        parse_dates=["Date"],
    ).set_index("Date")
    saved_test = saved.query("sample == 'test'").loc[features.index]
    parity = pd.DataFrame(
        {
            "check": [
                "feature_max_abs_difference",
                "probability_max_abs_difference",
                "saved_probability_max_abs_difference",
                "saved_position_mismatches",
            ],
            "value": [
                float(np.max(np.abs(batch_values - stream_values))),
                float(
                    np.max(
                        np.abs(batch_probability - stream_probability)
                    )
                ),
                float(
                    np.max(
                        np.abs(
                            saved_test["probability_up"].to_numpy()
                            - stream_probability
                        )
                    )
                ),
                int(
                    np.sum(
                        saved_test["dnn_position"].to_numpy()
                        != positions
                    )
                ),
            ],
        }
    )

    replay = PaperReplay(
        database_path,
        bundle.run_id,
        config.symbol,
        config.transaction_cost_one_way,
        max_drawdown=max_drawdown,
    )
    previous_time = features.index[0] - pd.Timedelta(days=1)
    previous_price = float(prices.loc[: features.index[0]].iloc[-2])
    for timestamp, realized_return, probability, position in zip(
        features.index,
        returns.to_numpy(),
        stream_probability,
        positions,
    ):
        current_price = float(prices.loc[timestamp])
        validate_observation(
            previous_time,
            timestamp,
            previous_price,
            current_price,
        )
        replay.process_bar(
            timestamp,
            current_price,
            float(realized_return),
            float(probability),
            int(position),
        )
        previous_time = timestamp
        previous_price = current_price
    telemetry, orders = replay.frames()
    replay.close()

    final_state = telemetry.iloc[-1]
    risk_orders = orders.query("reason == 'risk_flatten'")
    reconciliation = {
        "run_id": bundle.run_id,
        "bars_expected": len(features),
        "bars_recorded": len(telemetry),
        "final_position": int(final_state["position"]),
        "final_nav": float(final_state["nav"]),
        "maximum_drawdown": float(telemetry["drawdown"].min()),
        "halted": bool(final_state["halted"]),
        "risk_flatten_orders": len(risk_orders),
        "total_turnover": float(orders["turnover"].sum()),
        "parity_passed": bool(
            parity.loc[parity["check"] != "saved_position_mismatches", "value"]
            .lt(1e-6)
            .all()
            and parity.iloc[-1]["value"] == 0
        ),
    }
    failure_tests = _failure_tests(config.transaction_cost_one_way)
    return SessionThreeResults(
        database_path=Path(database_path),
        parity=parity,
        telemetry=telemetry,
        orders=orders,
        failure_tests=failure_tests,
        reconciliation=reconciliation,
    )


def persist_session_three(
    bundle: RunBundle,
    results: SessionThreeResults,
    code_commit: str | None = None,
) -> None:
    """Persist replay evidence without putting the database in Git."""
    bundle.validate(required_session=2)
    if 3 in bundle.manifest.get("completed_sessions", []):
        raise ValueError("Session 3 is already complete for this run.")
    destination = bundle.path / "session_3/paper_trading.db"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(results.database_path, destination)
    bundle.register_file("session_3/paper_trading.db")
    frames = {
        "session_3/parity.csv": results.parity,
        "session_3/telemetry.csv": results.telemetry,
        "session_3/orders.csv": results.orders,
        "session_3/failure_tests.csv": results.failure_tests,
    }
    for name, frame in frames.items():
        bundle.write_frame(name, frame)
    bundle.write_json(
        "session_3/reconciliation.json",
        results.reconciliation,
    )
    required = (
        "session_3/paper_trading.db",
        *frames.keys(),
        "session_3/reconciliation.json",
    )
    bundle.mark_session_complete(3, required, code_commit=code_commit)
