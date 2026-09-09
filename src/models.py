"""
PyTorch neural network architectures and baseline models for trading.

(c) Dr. Yves J. Hilpisch
The Python Quants GmbH | https://tpq.io
https://hilpisch.com
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class ModelConfig:
    """Serializable architecture specification for a trading network."""

    input_dim: int
    hidden_units: tuple[int, ...] = (64, 32)
    dropout_rate: float = 0.2
    use_batch_norm: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe configuration mapping."""
        payload = asdict(self)
        payload["hidden_units"] = list(self.hidden_units)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModelConfig":
        """Construct a model specification from serialized values."""
        return cls(
            input_dim=int(payload["input_dim"]),
            hidden_units=tuple(payload["hidden_units"]),
            dropout_rate=float(payload["dropout_rate"]),
            use_batch_norm=bool(payload["use_batch_norm"]),
        )


def get_device() -> torch.device:
    """Auto-detect best available compute accelerator (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class TradingDataset(Dataset):
    """PyTorch Dataset for financial feature matrices and target vectors."""

    def __init__(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray
    ):
        X_arr = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
        y_arr = y.to_numpy() if isinstance(y, pd.Series) else np.asarray(y)

        self.X = torch.tensor(X_arr, dtype=torch.float32)
        self.y = torch.tensor(y_arr, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class TradingDNN(nn.Module):
    """Deep Neural Network for directional market classification."""

    def __init__(
        self,
        input_dim: int,
        hidden_units: tuple[int, ...] = (64, 32),
        dropout_rate: float = 0.2,
        use_batch_norm: bool = False
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_units:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass outputting raw logits."""
        return self.network(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        r"""Forward pass outputting probabilities $p \in (0, 1)$."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)


def build_model(config: ModelConfig) -> TradingDNN:
    """Build exactly the architecture described by a model specification."""
    return TradingDNN(
        input_dim=config.input_dim,
        hidden_units=list(config.hidden_units),
        dropout_rate=config.dropout_rate,
        use_batch_norm=config.use_batch_norm,
    )


def save_model_checkpoint(
    path: str | Path,
    model: TradingDNN,
    config: ModelConfig,
    feature_names: list[str],
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    threshold: float,
    run_id: str,
) -> Path:
    """Atomically save the complete inference contract."""
    if len(feature_names) != config.input_dim:
        raise ValueError("Feature count does not match model input dimension.")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    payload = {
        "format_version": "1.0",
        "run_id": run_id,
        "model_config": config.as_dict(),
        "feature_names": feature_names,
        "scaler_mean": torch.as_tensor(
            np.asarray(scaler_mean, dtype=np.float32)
        ),
        "scaler_scale": torch.as_tensor(
            np.asarray(scaler_scale, dtype=np.float32)
        ),
        "threshold": float(threshold),
        "state_dict": model.state_dict(),
    }
    torch.save(payload, temporary)
    temporary.replace(destination)
    return destination


def load_model_checkpoint(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[TradingDNN, dict[str, Any]]:
    """Load and validate a complete inference contract."""
    try:
        payload = torch.load(
            path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        payload = torch.load(path, map_location=device)
    required = {
        "format_version",
        "run_id",
        "model_config",
        "feature_names",
        "scaler_mean",
        "scaler_scale",
        "threshold",
        "state_dict",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Checkpoint fields missing: {sorted(missing)}")
    config = ModelConfig.from_dict(payload["model_config"])
    if len(payload["feature_names"]) != config.input_dim:
        raise ValueError("Checkpoint feature contract is inconsistent.")
    model = build_model(config).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def save_ensemble_checkpoint(
    path: str | Path,
    models: list[TradingDNN],
    seeds: tuple[int, ...],
    config: ModelConfig,
    feature_names: list[str],
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    threshold: float,
    run_id: str,
) -> Path:
    """Atomically save a probability-averaging ensemble contract."""
    if not models:
        raise ValueError("An ensemble must contain at least one model.")
    if len(models) != len(seeds):
        raise ValueError("Every ensemble member must have one seed.")
    if len(feature_names) != config.input_dim:
        raise ValueError("Feature count does not match model input dimension.")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    payload = {
        "format_version": "2.0",
        "run_id": run_id,
        "model_config": config.as_dict(),
        "feature_names": feature_names,
        "scaler_mean": torch.as_tensor(
            np.asarray(scaler_mean, dtype=np.float32)
        ),
        "scaler_scale": torch.as_tensor(
            np.asarray(scaler_scale, dtype=np.float32)
        ),
        "threshold": float(threshold),
        "aggregation": "mean_probability",
        "seeds": list(seeds),
        "state_dicts": [model.state_dict() for model in models],
    }
    torch.save(payload, temporary)
    temporary.replace(destination)
    return destination


def load_ensemble_checkpoint(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[list[TradingDNN], dict[str, Any]]:
    """Load and validate a probability-averaging ensemble contract."""
    try:
        payload = torch.load(
            path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        payload = torch.load(path, map_location=device)
    required = {
        "format_version",
        "run_id",
        "model_config",
        "feature_names",
        "scaler_mean",
        "scaler_scale",
        "threshold",
        "aggregation",
        "seeds",
        "state_dicts",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Ensemble fields missing: {sorted(missing)}")
    if payload["aggregation"] != "mean_probability":
        raise ValueError("Unsupported ensemble aggregation rule.")
    if not payload["state_dicts"]:
        raise ValueError("The ensemble checkpoint has no members.")
    if len(payload["seeds"]) != len(payload["state_dicts"]):
        raise ValueError("Ensemble seeds and state dictionaries differ.")
    config = ModelConfig.from_dict(payload["model_config"])
    if len(payload["feature_names"]) != config.input_dim:
        raise ValueError("Checkpoint feature contract is inconsistent.")
    models = []
    for state_dict in payload["state_dicts"]:
        model = build_model(config).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)
    return models, payload


def ensemble_predict_proba(
    models: list[TradingDNN],
    features: torch.Tensor,
) -> torch.Tensor:
    """Average member probabilities for one feature tensor."""
    if not models:
        raise ValueError("An ensemble must contain at least one model.")
    with torch.no_grad():
        probabilities = [model.predict_proba(features) for model in models]
    return torch.stack(probabilities).mean(dim=0)


def train_trading_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: torch.device | None = None,
    save_path: str | Path | None = None,
    verbose: bool = True,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
) -> dict[str, list[float] | list[int]]:
    """Train PyTorch trading model with binary cross entropy with logits."""
    if device is None:
        device = get_device()

    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }
    best_val_loss = float("inf")
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_y)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == batch_y).sum().item()
            total += len(batch_y)

        train_loss = total_loss / total
        train_acc = correct / total
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        val_loss, val_acc = None, None
        if val_loader is not None:
            model.eval()
            v_total_loss = 0.0
            v_correct = 0
            v_total = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    logits = model(batch_x)
                    loss = criterion(logits, batch_y)
                    v_total_loss += loss.item() * len(batch_y)
                    preds = (torch.sigmoid(logits) > 0.5).float()
                    v_correct += (preds == batch_y).sum().item()
                    v_total += len(batch_y)

            val_loss = v_total_loss / v_total
            val_acc = v_correct / v_total
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            improved = (
                val_loss < best_val_loss - early_stopping_min_delta
            )
            if improved and save_path is not None:
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                best_state_dict = deepcopy(model.state_dict())
                torch.save(model.state_dict(), save_path)
            elif improved:
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                best_state_dict = deepcopy(model.state_dict())
            else:
                epochs_without_improvement += 1

        if verbose and (epoch % 10 == 0 or epoch == 1 or epoch == epochs):
            val_str = ""
            if val_loss is not None:
                val_str = (
                    f" | Val Loss: {val_loss:.4f} - "
                    f"Val Acc: {val_acc:.2%}"
                )
            print(
                f"Epoch [{epoch:03d}/{epochs:03d}] - "
                f"Train Loss: {train_loss:.4f} - "
                f"Train Acc: {train_acc:.2%}{val_str}"
            )

        if (
            val_loader is not None
            and early_stopping_patience is not None
            and epochs_without_improvement >= early_stopping_patience
        ):
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    epochs_trained = len(history["train_loss"])
    history["best_epoch"] = [best_epoch] * epochs_trained
    history["epochs_trained"] = [epochs_trained] * epochs_trained

    return history
