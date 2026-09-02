"""
ZeroMQ SUB client persisting real-time financial ticks into SQLite.

(c) Dr. Yves J. Hilpisch
The Python Quants GmbH | https://tpq.io
https://hilpisch.com
"""

from __future__ import annotations

import sqlite3
import zmq


def init_db(path: str = "ticks.db") -> sqlite3.Connection:
    """Create or open SQLite database for raw tick persistence."""
    conn = sqlite3.connect(path, check_same_thread=False)
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ticks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                time TEXT NOT NULL,
                symbol TEXT NOT NULL,
                price REAL NOT NULL
            )
        """)
    return conn


def run_tick_database(
    connect_addr: str = "tcp://127.0.0.1:5555",
    db_path: str = "ticks.db",
    max_ticks: int | None = None,
    timeout_ms: int = 500,
    max_idle_timeouts: int = 3
) -> None:
    """Subscribe to ZeroMQ tick stream and append each tick to SQLite."""
    conn = init_db(db_path)
    ctx = zmq.Context()
    socket = ctx.socket(zmq.SUB)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
    socket.connect(connect_addr)
    socket.setsockopt_string(zmq.SUBSCRIBE, "")

    print(
        f"[ZMQ DB Writer] Subscribed to {connect_addr}, "
        f"persisting to '{db_path}'..."
    )
    tick_count = 0
    idle_timeouts = 0
    try:
        while max_ticks is None or tick_count < max_ticks:
            try:
                tick = socket.recv_json()
                idle_timeouts = 0
            except zmq.Again:
                idle_timeouts += 1
                if idle_timeouts >= max_idle_timeouts:
                    print(
                        "[ZMQ DB Writer] Stream ended / idle timeout reached."
                    )
                    break
                continue
            except (zmq.ContextTerminated, zmq.ZMQError):
                break
            with conn:
                conn.execute(
                    "INSERT INTO ticks (time, symbol, price) VALUES (?, ?, ?)",
                    (tick["time"], tick["symbol"], float(tick["price"]))
                )
            tick_count += 1
            if tick_count % 50 == 0:
                print(
                    f"[ZMQ DB Writer] Persisted {tick_count} "
                    f"ticks to database."
                )
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        try:
            socket.close(0)
            conn.close()
            ctx.term()
        except Exception:
            pass
        print("[ZMQ DB Writer] Stopped and closed.")


if __name__ == "__main__":
    run_tick_database()
