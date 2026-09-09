"""
Canonical experiment configuration for the webinar series.

(c) Dr. Yves J. Hilpisch
The Python Quants GmbH | https://tpq.io
https://hilpisch.com
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


FX_COST_ONE_WAY = 0.00005


@dataclass(frozen=True)
class ExperimentConfig:
    """Shared financial, statistical, and execution assumptions."""

    symbol: str = "EURUSD"
    data_start: str = "2016-01-04"
    data_end: str = "2025-12-31"
    target_lags: int = 5
    factor_lags: int = 5
    train_ratio: float = 0.60
    validation_ratio: float = 0.20
    significance_level: float = 0.05
    periods_per_year: int = 252
    transaction_cost_one_way: float = FX_COST_ONE_WAY
    control_seed: int = 1
    random_seed: int = 42
    ensemble_members: int = 20
    schema_version: str = "1.0"

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable configuration mapping."""
        return asdict(self)

    def validate(self) -> None:
        """Validate assumptions shared across all sessions."""
        if self.target_lags < 1 or self.factor_lags < 1:
            raise ValueError("Lag counts must be positive.")
        if not 0.0 < self.significance_level < 1.0:
            raise ValueError("The significance level must lie in (0, 1).")
        if not 0.0 < self.train_ratio < 1.0:
            raise ValueError("The training ratio must lie in (0, 1).")
        if not 0.0 < self.validation_ratio < 1.0:
            raise ValueError("The validation ratio must lie in (0, 1).")
        if self.train_ratio + self.validation_ratio >= 1.0:
            raise ValueError("Training and validation must leave a test set.")
        if self.transaction_cost_one_way < 0.0:
            raise ValueError("Transaction costs cannot be negative.")
        if self.ensemble_members < 1:
            raise ValueError("The ensemble must contain at least one member.")
