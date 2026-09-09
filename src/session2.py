"""
Canonical Session 2 neural-network experiment and artifact persistence.

(c) Dr. Yves J. Hilpisch
The Python Quants GmbH | https://tpq.io
https://hilpisch.com
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from torch.utils.data import DataLoader

from src.artifacts import RunBundle
from src.backtest import run_vectorized_backtest
from src.config import ExperimentConfig
from src.data import create_lagged_features, load_eod_data
from src.models import (
    ModelConfig,
    TradingDNN,
    TradingDataset,
    build_model,
    ensemble_predict_proba,
    get_device,
    save_ensemble_checkpoint,
    train_trading_model,
)


@dataclass
class SessionTwoResults:
    """Artifacts produced by the canonical Session 2 experiment."""

    models: list[TradingDNN]
    seeds: tuple[int, ...]
    model_config: ModelConfig
    feature_names: list[str]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    threshold: float
    history: pd.DataFrame
    member_history: pd.DataFrame
    seed_metrics: pd.DataFrame
    threshold_results: pd.DataFrame
    strategy_metrics: pd.DataFrame
    predictions: pd.DataFrame


def _standardize(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Fit scaling parameters on training data and apply them unchanged."""
    mean = train.mean().to_numpy(dtype=np.float32)
    scale = train.std(ddof=0).to_numpy(dtype=np.float32)
    scale = np.where(scale == 0.0, 1.0, scale)
    columns = list(train.columns)

    def transform(frame: pd.DataFrame) -> pd.DataFrame:
        values = (frame.to_numpy() - mean) / scale
        return pd.DataFrame(values, index=frame.index, columns=columns)

    return (
        transform(train),
        transform(validation),
        transform(test),
        mean,
        scale,
    )


def _probabilities(
    model: TradingDNN,
    features: pd.DataFrame,
    device: torch.device,
) -> pd.Series:
    tensor = torch.tensor(
        features.to_numpy(),
        dtype=torch.float32,
        device=device,
    )
    model.eval()
    with torch.no_grad():
        values = torch.sigmoid(model(tensor)).cpu().numpy().ravel()
    return pd.Series(values, index=features.index, name="probability_up")


def _ensemble_probabilities(
    models: list[TradingDNN],
    features: pd.DataFrame,
    device: torch.device,
) -> pd.Series:
    """Average member probabilities for one chronological sample."""
    tensor = torch.tensor(
        features.to_numpy(),
        dtype=torch.float32,
        device=device,
    )
    values = ensemble_predict_proba(models, tensor).cpu().numpy().ravel()
    return pd.Series(values, index=features.index, name="probability_up")


def _threshold_positions(
    probabilities: pd.Series,
    threshold: float,
) -> pd.Series:
    positions = pd.Series(0.0, index=probabilities.index)
    positions.loc[probabilities > threshold] = 1.0
    positions.loc[probabilities < 1.0 - threshold] = -1.0
    return positions


def _metric_row(
    strategy: str,
    sample: str,
    returns: pd.Series,
    positions: pd.Series,
    config: ExperimentConfig,
) -> dict[str, float | str | int]:
    _, metrics = run_vectorized_backtest(
        returns,
        positions,
        tc=config.transaction_cost_one_way,
        periods_per_year=config.periods_per_year,
    )
    return {
        "strategy": strategy,
        "sample": sample,
        "observations": len(returns),
        "net_annual_return": metrics["Annualized Strategy Net Return"],
        "net_volatility": metrics["Annualized Volatility"],
        "net_sharpe": metrics["Sharpe Ratio (Strategy Net)"],
        "maximum_drawdown": metrics["Maximum Drawdown"],
        "turnover_units": metrics["Position Switches (Turnover Count)"],
    }


def _select_threshold(
    probabilities: pd.Series,
    returns: pd.Series,
    thresholds: tuple[float, ...],
    config: ExperimentConfig,
    strategy: str,
) -> tuple[float, pd.DataFrame]:
    """Select one threshold by validation net Sharpe only."""
    rows = []
    for threshold in thresholds:
        positions = _threshold_positions(probabilities, threshold)
        row = _metric_row(
            strategy,
            "validation",
            returns,
            positions,
            config,
        )
        row["threshold"] = threshold
        row["active_fraction"] = float((positions != 0.0).mean())
        rows.append(row)
    frame = pd.DataFrame(rows)
    selected = frame.sort_values(
        ["net_sharpe", "threshold"],
        ascending=[False, True],
    ).iloc[0]
    return float(selected["threshold"]), frame


def run_session_two(
    data_path: str | Path,
    config: ExperimentConfig | None = None,
    epochs: int = 20,
    batch_size: int = 64,
    thresholds: tuple[float, ...] = (0.50, 0.52, 0.55),
    hidden_units: tuple[int, ...] = (64, 32),
    dropout_rate: float = 0.2,
    ensemble_members: int | None = None,
    device: torch.device | None = None,
) -> SessionTwoResults:
    """Train an ensemble, select on validation, and evaluate test once."""
    active_config = config or ExperimentConfig()
    active_config.validate()
    member_count = (
        active_config.ensemble_members
        if ensemble_members is None
        else ensemble_members
    )
    if member_count < 1:
        raise ValueError("The ensemble must contain at least one member.")
    seeds = tuple(
        range(
            active_config.random_seed,
            active_config.random_seed + member_count,
        )
    )
    active_device = device or get_device()

    prices = load_eod_data(data_path, symbol=active_config.symbol)
    prices = prices.loc[active_config.data_start : active_config.data_end]
    features, target_return, target_direction = create_lagged_features(
        prices,
        lags=active_config.target_lags,
    )
    train_end = int(len(features) * active_config.train_ratio)
    validation_end = int(
        len(features)
        * (active_config.train_ratio + active_config.validation_ratio)
    )
    x_train = features.iloc[:train_end]
    x_validation = features.iloc[train_end:validation_end]
    x_test = features.iloc[validation_end:]
    y_train = target_direction.iloc[:train_end]
    y_validation = target_direction.iloc[train_end:validation_end]
    r_train = target_return.iloc[:train_end]
    r_validation = target_return.iloc[train_end:validation_end]
    r_test = target_return.iloc[validation_end:]
    (
        x_train_scaled,
        x_validation_scaled,
        x_test_scaled,
        scaler_mean,
        scaler_scale,
    ) = _standardize(x_train, x_validation, x_test)

    train_loader = DataLoader(
        TradingDataset(x_train_scaled, y_train),
        batch_size=batch_size,
        shuffle=False,
    )
    validation_loader = DataLoader(
        TradingDataset(x_validation_scaled, y_validation),
        batch_size=batch_size,
        shuffle=False,
    )
    model_config = ModelConfig(
        input_dim=x_train_scaled.shape[1],
        hidden_units=hidden_units,
        dropout_rate=dropout_rate,
    )
    models = []
    histories = []
    member_probabilities = {"validation": [], "test": []}
    seed_rows = []
    for member, seed in enumerate(seeds, start=1):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model = build_model(model_config)
        history = train_trading_model(
            model,
            train_loader,
            validation_loader,
            epochs=epochs,
            device=active_device,
            verbose=False,
        )
        member_frame = pd.DataFrame(history)
        member_frame.insert(0, "epoch", range(1, len(member_frame) + 1))
        member_frame.insert(0, "seed", seed)
        member_frame.insert(0, "member", member)
        histories.append(member_frame)
        validation_probability = _probabilities(
            model,
            x_validation_scaled,
            active_device,
        )
        test_probability = _probabilities(
            model,
            x_test_scaled,
            active_device,
        )
        member_probabilities["validation"].append(validation_probability)
        member_probabilities["test"].append(test_probability)
        member_threshold, _ = _select_threshold(
            validation_probability,
            r_validation,
            thresholds,
            active_config,
            "DNN Member",
        )
        for sample, returns, probability in (
            ("validation", r_validation, validation_probability),
            ("test", r_test, test_probability),
        ):
            positions = _threshold_positions(probability, member_threshold)
            row = _metric_row(
                "DNN Member",
                sample,
                returns,
                positions,
                active_config,
            )
            row["member"] = member
            row["seed"] = seed
            row["threshold"] = member_threshold
            row["active_fraction"] = float((positions != 0.0).mean())
            seed_rows.append(row)
        models.append(model)

    member_history = pd.concat(histories, ignore_index=True)
    history_frame = member_history.groupby("epoch", as_index=False)[
        ["train_loss", "val_loss", "train_acc", "val_acc"]
    ].mean()
    validation_probabilities = _ensemble_probabilities(
        models,
        x_validation_scaled,
        active_device,
    )
    test_probabilities = _ensemble_probabilities(
        models,
        x_test_scaled,
        active_device,
    )
    selected_threshold, threshold_results = _select_threshold(
        validation_probabilities,
        r_validation,
        thresholds,
        active_config,
        "DNN Ensemble",
    )
    probabilities = {
        "validation": validation_probabilities,
        "test": test_probabilities,
    }
    return_samples = {
        "validation": r_validation,
        "test": r_test,
    }
    feature_samples = {
        "validation": x_validation,
        "test": x_test,
    }

    ols = LinearRegression().fit(x_train_scaled, r_train)
    random_generator = np.random.default_rng(active_config.random_seed)
    metric_rows = []
    prediction_frames = []
    for sample in ("validation", "test"):
        sample_features = feature_samples[sample]
        sample_returns = return_samples[sample]
        sample_scaled = (
            x_validation_scaled if sample == "validation" else x_test_scaled
        )
        dnn_position = _threshold_positions(
            probabilities[sample],
            selected_threshold,
        )
        positions = {
            "DNN Ensemble": dnn_position,
            "OLS": pd.Series(
                np.sign(ols.predict(sample_scaled)),
                index=sample_returns.index,
            ),
            "Momentum": np.sign(sample_features["rolling_mom"]),
            "Random": pd.Series(
                random_generator.choice([-1.0, 1.0], len(sample_returns)),
                index=sample_returns.index,
            ),
            "Buy-and-hold": pd.Series(1.0, index=sample_returns.index),
        }
        for strategy, position in positions.items():
            metric_rows.append(
                _metric_row(
                    strategy,
                    sample,
                    sample_returns,
                    position,
                    active_config,
                )
            )
        predictions = pd.DataFrame(
            {
                "sample": sample,
                "target_return": sample_returns,
                "probability_up": probabilities[sample],
                "probability_std": pd.concat(
                    member_probabilities[sample], axis=1
                ).std(axis=1, ddof=0),
                "dnn_position": dnn_position,
                "ols_position": positions["OLS"],
                "momentum_position": positions["Momentum"],
                "random_position": positions["Random"],
            }
        )
        prediction_frames.append(predictions)

    return SessionTwoResults(
        models=models,
        seeds=seeds,
        model_config=model_config,
        feature_names=list(features.columns),
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        threshold=selected_threshold,
        history=history_frame,
        member_history=member_history,
        seed_metrics=pd.DataFrame(seed_rows),
        threshold_results=threshold_results,
        strategy_metrics=pd.DataFrame(metric_rows),
        predictions=pd.concat(prediction_frames).sort_index(),
    )


def persist_session_two(
    bundle: RunBundle,
    results: SessionTwoResults,
    code_commit: str | None = None,
) -> None:
    """Add the complete Session 2 inference contract to a run bundle."""
    bundle.validate(required_session=1)
    if 2 in bundle.manifest.get("completed_sessions", []):
        raise ValueError("Session 2 is already complete for this run.")
    checkpoint_path = bundle.path / "session_2/ensemble.pt"
    save_ensemble_checkpoint(
        checkpoint_path,
        results.models,
        results.seeds,
        results.model_config,
        results.feature_names,
        results.scaler_mean,
        results.scaler_scale,
        results.threshold,
        bundle.run_id,
    )
    bundle.register_file("session_2/ensemble.pt")
    bundle.write_json(
        "session_2/model_config.json",
        results.model_config.as_dict(),
    )
    bundle.write_json(
        "session_2/scaler.json",
        {
            "feature_names": results.feature_names,
            "mean": results.scaler_mean.tolist(),
            "scale": results.scaler_scale.tolist(),
        },
    )
    bundle.write_json(
        "session_2/selection.json",
        {
            "aggregation": "mean_probability",
            "ensemble_members": len(results.models),
            "seeds": list(results.seeds),
            "threshold": results.threshold,
            "selected_on": "validation",
        },
    )
    frames = {
        "session_2/training_history.csv": results.history,
        "session_2/member_training_history.csv": results.member_history,
        "session_2/seed_metrics.csv": results.seed_metrics,
        "session_2/threshold_results.csv": results.threshold_results,
        "session_2/strategy_metrics.csv": results.strategy_metrics,
        "session_2/predictions.csv": results.predictions.reset_index(),
    }
    for name, frame in frames.items():
        bundle.write_frame(name, frame)
    required = (
        "session_2/ensemble.pt",
        "session_2/model_config.json",
        "session_2/scaler.json",
        "session_2/selection.json",
        *frames.keys(),
    )
    bundle.mark_session_complete(2, required, code_commit=code_commit)
