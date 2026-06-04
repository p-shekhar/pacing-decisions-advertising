from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import DataCalibration


@dataclass(frozen=True)
class StreamingCase:
    case: str
    z_blocks: np.ndarray
    history_features: np.ndarray
    segment_table: pd.DataFrame
    campaign_table: pd.DataFrame
    response_scale: float
    member_scale: float
    metadata: dict[str, float]


def _resize_profile(values: tuple[float, ...] | None, length: int, fallback: np.ndarray) -> np.ndarray:
    if values is None or len(values) == 0:
        return fallback.astype(float)
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return fallback.astype(float)
    if arr.size < length:
        arr = np.resize(arr, length)
    else:
        arr = arr[:length]
    return arr.astype(float)


def generate_streaming_case(
    calibration: DataCalibration,
    n_blocks: int = 120,
    n_segments: int = 16,
    n_campaigns: int = 6,
    seed: int = 13,
) -> StreamingCase:
    """Generate the semi-synthetic streaming marketplace described in the paper.

    The raw public data calibrate base response, treatment lift, context count,
    inventory scale, demand volatility, and repeated-exposure/member-load scale.
    The generated forecast target z has three coordinates per block/segment:
    inventory mass, demand pressure, and quality-adjusted opportunity mass.
    Policy value and constraints are linear in this z vector, matching the
    affine downstream-inventory assumption used in the theory.
    """
    rng = np.random.default_rng(seed)
    n_segments = int(min(max(6, n_segments), max(6, calibration.n_contexts)))
    t = np.arange(n_blocks)
    seg = np.arange(n_segments)
    default_seasonal = 1.0 + 0.25 * np.sin(2 * np.pi * t / 24) + 0.15 * np.cos(2 * np.pi * t / 12)
    seasonal = _resize_profile(calibration.time_inventory_profile, n_blocks, default_seasonal)
    seasonal = seasonal / max(seasonal.mean(), 1e-9)
    default_popularity = rng.lognormal(mean=0.0, sigma=0.35, size=n_segments)
    segment_popularity = _resize_profile(calibration.context_popularity_profile, n_segments, default_popularity)
    segment_popularity /= segment_popularity.mean()
    inventory = (
        calibration.inventory_scale
        * seasonal[:, None]
        * segment_popularity[None, :]
        * rng.lognormal(0.0, 0.08 + 0.10 * calibration.demand_scale, size=(n_blocks, n_segments))
    )
    demand_pressure = (
        0.4
        + 0.35 * segment_popularity[None, :]
        + 0.15 * np.sin(2 * np.pi * (t[:, None] + seg[None, :]) / 18)
        + rng.normal(0, 0.04 + 0.05 * calibration.demand_scale, size=(n_blocks, n_segments))
    )
    demand_pressure = np.clip(demand_pressure, 0.02, None)
    quality = (
        calibration.base_rate
        + calibration.quality_scale * (0.5 + 0.5 * np.cos(2 * np.pi * (t[:, None] - seg[None, :]) / 30))
        + rng.normal(0, calibration.quality_scale * 0.15, size=(n_blocks, n_segments))
    )
    quality_profile = _resize_profile(calibration.context_quality_profile, n_segments, np.ones(n_segments))
    quality_profile = quality_profile / max(quality_profile.mean(), 1e-9)
    quality *= quality_profile[None, :]
    response_decay_strength = 0.0
    if calibration.case == "kuairand":
        response_decay_strength = 0.25 * calibration.repeated_exposure_scale
        decay = 1.0 - response_decay_strength * (1.0 - np.exp(-t / 24.0))
        quality *= decay[:, None]
    quality = np.clip(quality, 1e-4, None)

    z = np.stack([inventory, inventory * demand_pressure, inventory * quality], axis=-1)
    z_blocks = z.reshape(n_blocks, n_segments * 3)
    block_totals = z.sum(axis=1)
    lag_totals = np.vstack([block_totals[:1], block_totals[:-1]])
    trend = t / max(n_blocks - 1, 1)
    history_features = np.column_stack(
        [
            np.ones(n_blocks),
            trend,
            np.sin(2 * np.pi * t / 24),
            np.cos(2 * np.pi * t / 24),
            np.sin(2 * np.pi * t / 12),
            np.cos(2 * np.pi * t / 12),
            lag_totals[:, 0] / max(float(block_totals[:, 0].mean()), 1e-9),
            lag_totals[:, 1] / max(float(block_totals[:, 1].mean()), 1e-9),
            lag_totals[:, 2] / max(float(block_totals[:, 2].mean()), 1e-9),
        ]
    )

    segment_table = pd.DataFrame(
        {
            "segment_id": np.arange(n_segments),
            "segment_popularity": segment_popularity,
            "baseline_quality": quality.mean(axis=0),
        }
    )
    preferred = rng.integers(0, n_segments, size=n_campaigns)
    planning_inventory = float(inventory.mean(axis=0).sum())
    campaign_table = pd.DataFrame(
        {
            "campaign_id": np.arange(n_campaigns),
            "preferred_segment": preferred,
            "value_per_incremental": rng.uniform(8.0, 20.0, size=n_campaigns),
            "delivery_target": planning_inventory * rng.uniform(0.015, 0.035, size=n_campaigns),
            "budget": planning_inventory * rng.uniform(0.002, 0.006, size=n_campaigns),
            "max_member_load": planning_inventory * rng.uniform(0.035, 0.070, size=n_campaigns),
        }
    )
    response_scale = max(calibration.treatment_effect, calibration.outcome_scale * 0.05, 1e-4)
    member_scale = max(calibration.repeated_exposure_scale, 0.03)
    return StreamingCase(
        case=calibration.case,
        z_blocks=z_blocks,
        history_features=history_features,
        segment_table=segment_table,
        campaign_table=campaign_table,
        response_scale=response_scale,
        member_scale=member_scale,
        metadata={
            "n_blocks": float(n_blocks),
            "n_segments": float(n_segments),
            "n_campaigns": float(n_campaigns),
            "base_rate": calibration.base_rate,
            "treatment_effect": calibration.treatment_effect,
            "inventory_scale": calibration.inventory_scale,
            "response_decay_strength": float(response_decay_strength),
        },
    )
