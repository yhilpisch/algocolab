"""
ZeroMQ PUB server streaming real-time financial ticks.

(c) Dr. Yves J. Hilpisch
The Python Quants GmbH | https://tpq.io
https://hilpisch.com
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
import numpy as np
import zmq


def run_tick_server(
    bind_addr: str = "tcp://127.0.0.1:5555",
    symbol: str = "EURUSD",
    start_price: float = 1.1000,
    dt: float = 0.05,
    sigma: float = 0.0002,
    max_ticks: int | None = None
) -> None:
    """Publish financial market ticks as JSON on a ZeroMQ PUB socket."""
    ctx = zmq.Context()
    socket = ctx.socket(zmq.PUB)
    socket.setsockopt(zmq.LINGER, 0)
    try:
        socket.bind(bind_addr)
    except zmq.ZMQError as exc:
        print(
            f"[ZMQ Server Error] Address '{bind_addr}' is already in use.\n"
            f"Please terminate any existing tick server instances."
        )
        socket.close(0)
        ctx.term()
        return

    price = start_price
    rng = np.random.default_rng()
    tick_count = 0

    print(
        f"[ZMQ Server] Publishing {symbol} ticks on {bind_addr} "
        f"(dt={dt}s)..."
    )
    try:
        while max_ticks is None or tick_count < max_ticks:
            shock = rng.normal(0.0, sigma * price)
            price = max(1.0, price + shock)
            payload = {
                "time": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "price": round(float(price), 4)
            }
            try:
                socket.send_json(payload)
            except (zmq.ContextTerminated, zmq.ZMQError):
                break
            tick_count += 1
            time.sleep(dt)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        try:
            socket.close(0)
            ctx.term()
        except Exception:
            pass
        print("[ZMQ Server] Stopped and socket closed.")


if __name__ == "__main__":
    run_tick_server()
