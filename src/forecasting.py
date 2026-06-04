from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd


@dataclass
class ForecastResult:
    model: str
    predictions: np.ndarray
    mse: float
    notes: str
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    band_radius: float | None = None


def _progress(message: str, verbose: bool = True) -> None:
    if verbose:
        print(f"[forecast] {message}", flush=True)


def split_blocks(n_blocks: int, train_frac: float = 0.5, cal_frac: float = 0.25) -> dict[str, np.ndarray]:
    idx = np.arange(n_blocks)
    n_train = max(4, int(n_blocks * train_frac))
    n_cal = max(4, int(n_blocks * cal_frac))
    return {
        "train": idx[:n_train],
        "cal": idx[n_train : n_train + n_cal],
        "test": idx[n_train + n_cal :],
    }


def _time_features(n: int) -> np.ndarray:
    t = np.arange(n, dtype=float)
    x = np.column_stack(
        [
            np.ones(n),
            t / max(n - 1, 1),
            np.sin(2 * np.pi * t / 24),
            np.cos(2 * np.pi * t / 24),
            np.sin(2 * np.pi * t / 12),
            np.cos(2 * np.pi * t / 12),
        ]
    )
    return x


def _forecast_features(n: int, history: np.ndarray | None = None) -> np.ndarray:
    if history is None:
        return _time_features(n)
    h = np.asarray(history, dtype=float)
    if h.shape[0] != n:
        raise ValueError("history must have one row per forecast block.")
    fallback = _time_features(n)
    h = np.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
    return np.column_stack([fallback, h])


def _fit_ridge(x: np.ndarray, y: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    xtx = x.T @ x + lam * np.eye(x.shape[1])
    return np.linalg.solve(xtx, x.T @ y)


def naive_forecast(z: np.ndarray) -> np.ndarray:
    pred = np.empty_like(z)
    pred[0] = z[0]
    pred[1:] = z[:-1]
    return pred


def seasonal_ridge_forecast(
    z: np.ndarray,
    train_idx: np.ndarray,
    history: np.ndarray | None = None,
) -> np.ndarray:
    x = _forecast_features(len(z), history)
    beta = _fit_ridge(x[train_idx], z[train_idx], lam=1e-2)
    pred = x @ beta
    return np.clip(pred, 0, None)


def random_feature_forecast(
    z: np.ndarray,
    train_idx: np.ndarray,
    history: np.ndarray | None = None,
    n_features: int = 96,
    seed: int = 7,
) -> np.ndarray:
    """A nonlinear random-feature forecaster used as a light neural-style baseline."""
    rng = np.random.default_rng(seed)
    x = _forecast_features(len(z), history)
    weights = rng.normal(0, 1.0, size=(x.shape[1], n_features))
    bias = rng.normal(0, 0.5, size=n_features)
    hidden = np.tanh(x @ weights + bias)
    design = np.column_stack([x, hidden])
    beta = _fit_ridge(design[train_idx], z[train_idx], lam=1e-1)
    pred = design @ beta
    return np.clip(pred, 0, None)


def numpy_mlp_forecast(
    z: np.ndarray,
    train_idx: np.ndarray,
    history: np.ndarray | None = None,
    hidden: int = 48,
    epochs: int = 600,
    lr: float = 0.015,
    seed: int = 11,
    verbose: bool = False,
) -> np.ndarray:
    """Small two-layer neural forecaster trained in numpy.

    The implementation is compact and reproducible. It is not
    meant to beat specialized time-series libraries; it supplies the paper's
    neural forecasting layer while leaving the decision calibration unchanged.
    """
    rng = np.random.default_rng(seed)
    x = _forecast_features(len(z), history)
    y = z.copy()
    y_scale = np.maximum(y[train_idx].std(axis=0, keepdims=True), 1e-6)
    y_mean = y[train_idx].mean(axis=0, keepdims=True)
    ys = (y - y_mean) / y_scale
    w1 = rng.normal(0, 0.4, size=(x.shape[1], hidden))
    b1 = np.zeros(hidden)
    w2 = rng.normal(0, 0.05, size=(hidden, z.shape[1]))
    b2 = np.zeros(z.shape[1])
    xt = x[train_idx]
    yt = ys[train_idx]
    report_every = max(1, epochs // 5)
    _progress(f"numpy_mlp training start: epochs={epochs}, hidden={hidden}", verbose)
    for epoch in range(epochs):
        h = np.tanh(xt @ w1 + b1)
        out = h @ w2 + b2
        err = (out - yt) / len(train_idx)
        grad_w2 = h.T @ err
        grad_b2 = err.sum(axis=0)
        grad_h = err @ w2.T
        grad_a = grad_h * (1 - h**2)
        grad_w1 = xt.T @ grad_a
        grad_b1 = grad_a.sum(axis=0)
        w2 -= lr * grad_w2
        b2 -= lr * grad_b2
        w1 -= lr * grad_w1
        b1 -= lr * grad_b1
        if verbose and ((epoch + 1) % report_every == 0 or epoch == 0):
            loss = float(np.mean((out - yt) ** 2))
            _progress(f"numpy_mlp epoch {epoch + 1}/{epochs}, train_mse={loss:.6g}", verbose)
    pred = np.tanh(x @ w1 + b1) @ w2 + b2
    pred = pred * y_scale + y_mean
    return np.clip(pred, 0, None)


def gradient_boosted_forecast(
    z: np.ndarray,
    train_idx: np.ndarray,
    history: np.ndarray | None = None,
    seed: int = 29,
    verbose: bool = False,
) -> np.ndarray:
    """Histogram gradient-boosted forecaster trained on streaming history features."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    x = _forecast_features(len(z), history)
    pred = np.zeros_like(z, dtype=float)
    report_every = max(1, z.shape[1] // 6)
    _progress(
        f"gradient_boosted training start: targets={z.shape[1]}, train_blocks={len(train_idx)}",
        verbose,
    )
    for j in range(z.shape[1]):
        model = HistGradientBoostingRegressor(
            max_iter=180,
            learning_rate=0.05,
            l2_regularization=0.05,
            random_state=seed + j,
        )
        model.fit(x[train_idx], z[train_idx, j])
        pred[:, j] = model.predict(x)
        if verbose and ((j + 1) % report_every == 0 or j == 0 or j + 1 == z.shape[1]):
            _progress(f"gradient_boosted fitted target {j + 1}/{z.shape[1]}", verbose)
    return np.clip(pred, 0, None)


def _resolve_torch_device(torch_module, device: str = "auto"):
    if device != "auto":
        return torch_module.device(device)
    if torch_module.cuda.is_available():
        return torch_module.device("cuda")
    if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
        return torch_module.device("mps")
    return torch_module.device("cpu")


def torch_gru_forecast(
    z: np.ndarray,
    train_idx: np.ndarray,
    history: np.ndarray | None = None,
    lookback: int = 8,
    hidden: int = 64,
    epochs: int = 250,
    lr: float = 0.01,
    seed: int = 17,
    device: str = "auto",
    verbose: bool = False,
) -> np.ndarray:
    """PyTorch GRU one-step forecaster for streaming inventory states.

    The model predicts the next inventory vector from the previous `lookback`
    realized vectors. It is used to enrich the forecasting comparison while
    leaving the decision-calibrated conformal layer unchanged.
    """
    import torch
    from torch import nn

    torch.manual_seed(seed)
    torch_device = _resolve_torch_device(torch, device)
    _progress(f"torch_gru using device={torch_device}", verbose)
    if torch_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision("high")
    z = np.asarray(z, dtype=np.float32)
    n, dim = z.shape
    hx = _forecast_features(n, history).astype(np.float32)
    train_set = set(int(i) for i in train_idx)
    y_mean = z[train_idx].mean(axis=0, keepdims=True)
    y_scale = np.maximum(z[train_idx].std(axis=0, keepdims=True), 1e-6)
    zs = (z - y_mean) / y_scale
    h_mean = hx[train_idx].mean(axis=0, keepdims=True)
    h_scale = np.maximum(hx[train_idx].std(axis=0, keepdims=True), 1e-6)
    hs = (hx - h_mean) / h_scale
    seq = np.concatenate([zs, hs], axis=1)

    x_rows = []
    y_rows = []
    for t in range(lookback, n):
        if t in train_set:
            x_rows.append(seq[t - lookback : t])
            y_rows.append(zs[t])
    if len(x_rows) < 4:
        _progress("torch_gru skipped: fewer than 4 training windows", verbose)
        return naive_forecast(z)

    x_train = torch.tensor(np.stack(x_rows), dtype=torch.float32, device=torch_device)
    y_train = torch.tensor(np.stack(y_rows), dtype=torch.float32, device=torch_device)

    input_dim = seq.shape[1]

    class GRUForecaster(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
            super().__init__()
            self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, output_dim),
            )

        def forward(self, x):
            out, _ = self.gru(x)
            return self.head(out[:, -1, :])

    model = GRUForecaster(input_dim, hidden, dim).to(torch_device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()
    model.train()
    report_every = max(1, epochs // 5)
    _progress(
        f"torch_gru training start: epochs={epochs}, windows={len(x_rows)}, lookback={lookback}",
        verbose,
    )
    for epoch in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x_train), y_train)
        loss.backward()
        opt.step()
        if verbose and ((epoch + 1) % report_every == 0 or epoch == 0):
            _progress(f"torch_gru epoch {epoch + 1}/{epochs}, loss={float(loss.item()):.6g}", verbose)

    pred = np.empty_like(zs)
    pred[:lookback] = zs[:lookback]
    model.eval()
    _progress(f"torch_gru inference start: blocks={n - lookback}", verbose)
    with torch.no_grad():
        for t in range(lookback, n):
            x = torch.tensor(
                seq[t - lookback : t][None, :, :],
                dtype=torch.float32,
                device=torch_device,
            )
            pred[t] = model(x).cpu().numpy()[0]
    pred = pred * y_scale + y_mean
    return np.clip(pred.astype(float), 0, None)


def torch_transformer_forecast(
    z: np.ndarray,
    train_idx: np.ndarray,
    history: np.ndarray | None = None,
    lookback: int = 12,
    d_model: int = 96,
    nhead: int = 4,
    num_layers: int = 2,
    epochs: int = 350,
    lr: float = 0.006,
    seed: int = 23,
    device: str = "auto",
    verbose: bool = False,
) -> np.ndarray:
    """PyTorch temporal Transformer forecaster for streaming inventory states.

    The model uses a Transformer encoder over the previous `lookback` realized
    inventory vectors and predicts the next inventory vector. This is the
    strongest deep baseline in the workflow and is intentionally evaluated by
    the same decision-calibrated radius as every other forecasting model.
    """
    import torch
    from torch import nn

    torch.manual_seed(seed)
    torch_device = _resolve_torch_device(torch, device)
    _progress(f"torch_transformer using device={torch_device}", verbose)
    if torch_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision("high")
    z = np.asarray(z, dtype=np.float32)
    n, dim = z.shape
    hx = _forecast_features(n, history).astype(np.float32)
    train_set = set(int(i) for i in train_idx)
    y_mean = z[train_idx].mean(axis=0, keepdims=True)
    y_scale = np.maximum(z[train_idx].std(axis=0, keepdims=True), 1e-6)
    zs = (z - y_mean) / y_scale
    h_mean = hx[train_idx].mean(axis=0, keepdims=True)
    h_scale = np.maximum(hx[train_idx].std(axis=0, keepdims=True), 1e-6)
    hs = (hx - h_mean) / h_scale
    seq = np.concatenate([zs, hs], axis=1)

    x_rows = []
    y_rows = []
    for t in range(lookback, n):
        if t in train_set:
            x_rows.append(seq[t - lookback : t])
            y_rows.append(zs[t])
    if len(x_rows) < 4:
        _progress("torch_transformer skipped: fewer than 4 training windows", verbose)
        return naive_forecast(z)

    x_train = torch.tensor(np.stack(x_rows), dtype=torch.float32, device=torch_device)
    y_train = torch.tensor(np.stack(y_rows), dtype=torch.float32, device=torch_device)

    class TemporalTransformer(nn.Module):
        def __init__(
            self,
            input_dim: int,
            model_dim: int,
            heads: int,
            layers: int,
            output_dim: int,
            window: int,
        ) -> None:
            super().__init__()
            self.input = nn.Linear(input_dim, model_dim)
            self.position = nn.Parameter(torch.zeros(1, window, model_dim))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=heads,
                dim_feedforward=4 * model_dim,
                dropout=0.05,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
            self.head = nn.Sequential(
                nn.LayerNorm(model_dim),
                nn.Linear(model_dim, model_dim),
                nn.GELU(),
                nn.Linear(model_dim, output_dim),
            )

        def forward(self, x):
            h = self.input(x) + self.position[:, : x.shape[1], :]
            encoded = self.encoder(h)
            return self.head(encoded[:, -1, :])

    model = TemporalTransformer(seq.shape[1], d_model, nhead, num_layers, dim, lookback).to(torch_device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()
    model.train()
    report_every = max(1, epochs // 5)
    _progress(
        (
            "torch_transformer training start: "
            f"epochs={epochs}, windows={len(x_rows)}, lookback={lookback}, "
            f"d_model={d_model}, layers={num_layers}"
        ),
        verbose,
    )
    for epoch in range(epochs):
        opt.zero_grad(set_to_none=True)
        pred_train = model(x_train)
        loss = loss_fn(pred_train, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        if epoch == int(epochs * 0.65):
            for group in opt.param_groups:
                group["lr"] *= 0.35
        if verbose and ((epoch + 1) % report_every == 0 or epoch == 0):
            _progress(
                f"torch_transformer epoch {epoch + 1}/{epochs}, loss={float(loss.item()):.6g}",
                verbose,
            )

    pred = np.empty_like(zs)
    pred[:lookback] = zs[:lookback]
    model.eval()
    _progress(f"torch_transformer inference start: blocks={n - lookback}", verbose)
    with torch.no_grad():
        for t in range(lookback, n):
            x = torch.tensor(
                seq[t - lookback : t][None, :, :],
                dtype=torch.float32,
                device=torch_device,
            )
            pred[t] = model(x).cpu().numpy()[0]
    pred = pred * y_scale + y_mean
    return np.clip(pred.astype(float), 0, None)


def _prediction_band(
    z: np.ndarray,
    pred: np.ndarray,
    train_idx: np.ndarray,
    alpha: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Construct a calibrated max-coordinate residual band around a point forecast."""
    residual_scores = np.max(np.abs(z[train_idx] - pred[train_idx]), axis=1)
    radius = conformal_quantile(residual_scores, alpha=alpha)
    lower = np.clip(pred - radius, 0, None)
    upper = pred + radius
    return lower, upper, radius


def fit_all_forecasters(
    z: np.ndarray,
    train_idx: np.ndarray,
    history: np.ndarray | None = None,
    alpha: float = 0.1,
    model_names: list[str] | tuple[str, ...] | set[str] | None = None,
    verbose: bool = True,
) -> dict[str, ForecastResult]:
    model_specs = [
        ("point_naive", lambda: naive_forecast(z), "previous-block point forecast"),
        (
            "seasonal_ridge",
            lambda: seasonal_ridge_forecast(z, train_idx, history),
            "linear ridge forecaster on streaming history features",
        ),
        (
            "random_feature",
            lambda: random_feature_forecast(z, train_idx, history),
            "nonlinear random-feature forecaster on streaming history features",
        ),
        (
            "gradient_boosted",
            lambda: gradient_boosted_forecast(z, train_idx, history, verbose=verbose),
            "histogram gradient-boosted tree forecaster on streaming history features",
        ),
        (
            "numpy_mlp",
            lambda: numpy_mlp_forecast(z, train_idx, history, verbose=verbose),
            "two-layer numpy neural forecaster on streaming history features",
        ),
        (
            "torch_gru",
            lambda: torch_gru_forecast(z, train_idx, history, verbose=verbose),
            "PyTorch GRU sequence forecaster over streaming history states",
        ),
        (
            "torch_transformer",
            lambda: torch_transformer_forecast(z, train_idx, history, verbose=verbose),
            "PyTorch temporal Transformer encoder over streaming history states",
        ),
    ]
    if model_names is not None:
        wanted = set(model_names)
        model_specs = [spec for spec in model_specs if spec[0] in wanted]
        missing = wanted - {spec[0] for spec in model_specs}
        if missing:
            raise ValueError(f"Unknown forecaster(s): {sorted(missing)}")
    out = {}
    _progress(
        f"fitting {len(model_specs)} forecasters: blocks={len(z)}, dim={z.shape[1]}, train_blocks={len(train_idx)}",
        verbose,
    )
    for name, fit_fn, notes in model_specs:
        start = perf_counter()
        _progress(f"{name} start", verbose)
        pred = fit_fn()
        mse = float(np.mean((z[train_idx] - pred[train_idx]) ** 2))
        lower, upper, band_radius = _prediction_band(z, pred, train_idx, alpha=alpha)
        out[name] = ForecastResult(name, pred, mse, notes, lower, upper, band_radius)
        _progress(
            f"{name} done in {perf_counter() - start:.1f}s, train_mse={mse:.6g}, band_radius={band_radius:.6g}",
            verbose,
        )
    return out


def conformal_quantile(scores: np.ndarray, alpha: float = 0.1) -> float:
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return 0.0
    k = int(np.ceil((scores.size + 1) * (1 - alpha)))
    k = min(max(k, 1), scores.size)
    return float(np.sort(scores)[k - 1])


def forecast_summary_table(results: dict[str, ForecastResult], z: np.ndarray, idx: np.ndarray) -> pd.DataFrame:
    rows = []
    for name, res in results.items():
        rows.append(
            {
                "model": name,
                "mse": float(np.mean((z[idx] - res.predictions[idx]) ** 2)),
                "mae": float(np.mean(np.abs(z[idx] - res.predictions[idx]))),
                "band_radius": float(res.band_radius) if res.band_radius is not None else np.nan,
                "band_coverage": float(
                    np.mean(np.all((z[idx] >= res.lower[idx]) & (z[idx] <= res.upper[idx]), axis=1))
                )
                if res.lower is not None and res.upper is not None
                else np.nan,
                "mean_band_width": float(np.mean(res.upper[idx] - res.lower[idx]))
                if res.lower is not None and res.upper is not None
                else np.nan,
                "notes": res.notes,
            }
        )
    return pd.DataFrame(rows).sort_values("mse")
