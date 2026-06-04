from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tarfile
import io
from zipfile import ZipFile

import numpy as np
import pandas as pd

from .config import DATA_DIR


@dataclass(frozen=True)
class DataCalibration:
    case: str
    n_rows: int
    base_rate: float
    treatment_effect: float
    outcome_scale: float
    inventory_scale: float
    quality_scale: float
    demand_scale: float
    repeated_exposure_scale: float
    n_contexts: int
    time_inventory_profile: tuple[float, ...] | None = None
    context_popularity_profile: tuple[float, ...] | None = None
    context_quality_profile: tuple[float, ...] | None = None


@dataclass(frozen=True)
class CausalResponseSummary:
    case: str
    predicted_effect: float
    doubly_robust_effect: float
    robust_effect: float
    doubly_robust_se: float
    propensity_min: float
    propensity_max: float
    notes: str


def dataset_readiness() -> pd.DataFrame:
    rows = []
    candidates = {
        "criteo_uplift": DATA_DIR / "criteo" / "criteo-research-uplift-v2.1.csv.gz",
        "kuairec_processed": DATA_DIR / "processed" / "kuairec_user_day_panel_sample.parquet",
        "kuairec_long_term": DATA_DIR / "processed" / "kuairec_long_term_estimand_panel.parquet",
        "kuairand_archive": DATA_DIR / "KuaiRand" / "10439422.zip",
        "open_bandit_random_men": DATA_DIR / "processed" / "open_bandit_random_men_sample.parquet",
    }
    for name, path in candidates.items():
        rows.append(
            {
                "dataset": name,
                "path": str(path),
                "exists": path.exists(),
                "size_mb": round(path.stat().st_size / 1_000_000, 3) if path.exists() else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _detect_column(columns: list[str], candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for key in candidates:
        if key.lower() in lower:
            return lower[key.lower()]
    for key in candidates:
        for col in columns:
            if key.lower() in col.lower():
                return col
    return None


def load_criteo(max_rows: int = 500_000) -> pd.DataFrame:
    path = DATA_DIR / "criteo" / "criteo-research-uplift-v2.1.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"Missing Criteo data at {path}")
    target_per_arm = max(1, max_rows // 2)
    treated_parts = []
    control_parts = []
    treatment_col = None
    outcome_col = None
    for chunk in pd.read_csv(path, chunksize=250_000):
        if treatment_col is None or outcome_col is None:
            treatment_col = _detect_column(chunk.columns.tolist(), ["treatment", "treated", "exposure"])
            outcome_col = _detect_column(chunk.columns.tolist(), ["conversion", "visit", "outcome"])
            if treatment_col is None or outcome_col is None:
                raise ValueError("Could not detect treatment/outcome columns in Criteo data.")
        treatment = pd.to_numeric(chunk[treatment_col], errors="coerce").fillna(0)
        treated_chunk = chunk[treatment > 0.5]
        control_chunk = chunk[treatment <= 0.5]
        if len(treated_chunk) and sum(len(x) for x in treated_parts) < target_per_arm:
            need = target_per_arm - sum(len(x) for x in treated_parts)
            treated_parts.append(treated_chunk.head(need))
        if len(control_chunk) and sum(len(x) for x in control_parts) < target_per_arm:
            need = target_per_arm - sum(len(x) for x in control_parts)
            control_parts.append(control_chunk.head(need))
        if sum(len(x) for x in treated_parts) >= target_per_arm and sum(len(x) for x in control_parts) >= target_per_arm:
            break
    if treatment_col is None or outcome_col is None:
        raise ValueError("Could not detect treatment/outcome columns in Criteo data.")
    parts = treated_parts + control_parts
    if not parts:
        raise ValueError("Could not load Criteo rows.")
    df = pd.concat(parts, ignore_index=True).head(max_rows)
    feature_cols = [c for c in df.columns if c not in {treatment_col, outcome_col}]
    if not feature_cols:
        feature_cols = [outcome_col]
    out = pd.DataFrame(
        {
            "unit_id": np.arange(len(df)),
            "treatment": df[treatment_col].astype(float),
            "outcome": df[outcome_col].astype(float),
        }
    )
    feature_frame = df[feature_cols[: min(8, len(feature_cols))]].copy()
    context_codes = []
    for col in feature_cols[: min(4, len(feature_cols))]:
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().sum() > 0 and numeric.nunique(dropna=True) > 4:
            ranks = numeric.rank(method="first", pct=True).fillna(0.0)
            code = np.floor(np.clip(ranks.to_numpy(), 0, 0.999) * 4).astype(int)
        else:
            code = pd.factorize(df[col].fillna("missing").astype(str))[0] % 4
        context_codes.append(code)
    if context_codes:
        context = np.zeros(len(df), dtype=int)
        for j, code in enumerate(context_codes):
            context += code * (4**j)
        out["context_id"] = context % 64
    else:
        out["context_id"] = np.arange(len(out)) % 64
    out["time_index"] = np.arange(len(out)) // max(1, len(out) // 96)
    out["quality_proxy"] = feature_frame.apply(pd.to_numeric, errors="coerce").fillna(0).mean(axis=1)
    q = out["quality_proxy"]
    out["quality_proxy"] = (q - q.min()) / (q.max() - q.min() + 1e-9)
    return out


def calibrate_criteo(max_rows: int = 500_000) -> DataCalibration:
    df = load_criteo(max_rows=max_rows)
    treated = df["treatment"] > 0.5
    y1 = df.loc[treated, "outcome"].mean()
    y0 = df.loc[~treated, "outcome"].mean()
    effect = float(y1 - y0) if np.isfinite(y1 - y0) else 0.0
    base = float(df["outcome"].mean())
    contexts = int(df["context_id"].nunique())
    time_counts = df.groupby("time_index").size().astype(float)
    time_profile = (time_counts / max(time_counts.mean(), 1e-9)).to_numpy()
    context_counts = df.groupby("context_id").size().astype(float).sort_index()
    context_popularity = (context_counts / max(context_counts.mean(), 1e-9)).to_numpy()
    context_quality = df.groupby("context_id")["quality_proxy"].mean().sort_index().to_numpy()
    return DataCalibration(
        case="criteo",
        n_rows=len(df),
        base_rate=max(base, 1e-4),
        treatment_effect=effect,
        outcome_scale=max(abs(effect), base, 1e-4),
        inventory_scale=float(len(df) / 96),
        quality_scale=float(df["quality_proxy"].std() + 1e-3),
        demand_scale=float(df.groupby("time_index").size().std() / max(1, df.groupby("time_index").size().mean())),
        repeated_exposure_scale=0.05,
        n_contexts=contexts,
        time_inventory_profile=tuple(float(x) for x in time_profile),
        context_popularity_profile=tuple(float(x) for x in context_popularity),
        context_quality_profile=tuple(float(x) for x in context_quality),
    )


def _load_first_existing(paths: list[Path], max_rows: int) -> pd.DataFrame:
    for path in paths:
        if path.exists():
            if path.suffix == ".parquet":
                return pd.read_parquet(path).head(max_rows)
            if path.suffix == ".csv":
                return pd.read_csv(path, nrows=max_rows)
    raise FileNotFoundError("None of the requested KuaiRand/KuaiRec processed files exist.")


def load_kuairand(max_rows: int = 500_000) -> pd.DataFrame:
    path = DATA_DIR / "KuaiRand" / "10439422.zip"
    if path.exists():
        with ZipFile(path) as zf:
            with zf.open("KuaiRand-Pure.tar.gz") as nested:
                with tarfile.open(fileobj=nested, mode="r|gz") as tf:
                    for member in tf:
                        if member.name.endswith("log_random_4_22_to_5_08_pure.csv"):
                            fh = tf.extractfile(member)
                            if fh is None:
                                break
                            df = pd.read_csv(io.BytesIO(fh.read()), nrows=max_rows)
                            out = pd.DataFrame(
                                {
                                    "unit_id": np.arange(len(df)),
                                    "user_id": df["user_id"],
                                    "item_id": df["video_id"],
                                    "outcome": (
                                        0.6 * pd.to_numeric(df["is_click"], errors="coerce").fillna(0)
                                        + 0.3 * pd.to_numeric(df["long_view"], errors="coerce").fillna(0)
                                        + 0.1 * pd.to_numeric(df["is_like"], errors="coerce").fillna(0)
                                    ),
                                    "time_index": pd.factorize(df["date"].astype(str) + "_" + df["hourmin"].astype(str))[0],
                                    "context_id": pd.factorize(df["video_id"])[0] % 64,
                                    "treatment": pd.to_numeric(df["is_rand"], errors="coerce").fillna(1),
                                }
                            )
                            return out
    return load_kuairec_processed(max_rows=max_rows)


def load_kuairec_processed(max_rows: int = 500_000) -> pd.DataFrame:
    paths = [
        DATA_DIR / "processed" / "kuairec_user_day_panel_sample.parquet",
        DATA_DIR / "processed" / "kuairec_long_term_estimand_panel.parquet",
        DATA_DIR / "processed" / "kuairec_small_interactions_sample.parquet",
    ]
    df = _load_first_existing(paths, max_rows=max_rows)
    columns = df.columns.tolist()
    user_col = _detect_column(columns, ["user_id", "user", "uid"])
    item_col = _detect_column(columns, ["video_id", "item_id", "item", "photo_id"])
    time_col = _detect_column(columns, ["date", "day", "time", "timestamp"])
    outcome_col = _detect_column(
        columns,
        ["watch_ratio", "watch_time", "play_time", "is_click", "click", "like", "outcome"],
    )
    if outcome_col is None:
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric:
            raise ValueError("Could not find a numeric outcome-like column in KuaiRec data.")
        outcome_col = numeric[0]
    out = pd.DataFrame(
        {
            "unit_id": np.arange(len(df)),
            "user_id": df[user_col] if user_col else np.arange(len(df)),
            "item_id": df[item_col] if item_col else np.arange(len(df)) % 100,
            "outcome": pd.to_numeric(df[outcome_col], errors="coerce").fillna(0.0),
        }
    )
    if time_col:
        time_values = pd.factorize(df[time_col])[0]
    else:
        time_values = np.arange(len(df)) // max(1, len(df) // 96)
    out["time_index"] = time_values
    out["context_id"] = pd.factorize(out["item_id"])[0] % 64
    y = out["outcome"].astype(float)
    if y.max() > y.min():
        out["outcome"] = (y - y.min()) / (y.max() - y.min())
    out["treatment"] = 1.0
    return out


def load_kuairec(max_rows: int = 500_000) -> pd.DataFrame:
    """Backward-compatible alias. The paper-facing path uses KuaiRand first."""
    return load_kuairand(max_rows=max_rows)


def calibrate_kuairand(max_rows: int = 500_000) -> DataCalibration:
    df = load_kuairand(max_rows=max_rows)
    repeated = df.groupby("user_id").size()
    repeat_scale = float(np.clip((repeated.mean() - 1) / max(repeated.mean(), 1), 0.02, 0.9))
    base = float(df["outcome"].mean())
    by_time = df.groupby("time_index").size()
    time_profile = (by_time.astype(float) / max(by_time.mean(), 1e-9)).to_numpy()
    context_counts = df.groupby("context_id").size().astype(float).sort_index()
    context_popularity = (context_counts / max(context_counts.mean(), 1e-9)).to_numpy()
    context_quality = df.groupby("context_id")["outcome"].mean().sort_index().to_numpy()
    return DataCalibration(
        case="kuairand",
        n_rows=len(df),
        base_rate=max(base, 1e-4),
        treatment_effect=max(0.02 * base, 1e-4),
        outcome_scale=max(base, 1e-4),
        inventory_scale=float(len(df) / max(1, df["time_index"].nunique())),
        quality_scale=float(df["outcome"].std() + 1e-3),
        demand_scale=float(by_time.std() / max(1, by_time.mean())),
        repeated_exposure_scale=repeat_scale,
        n_contexts=int(df["context_id"].nunique()),
        time_inventory_profile=tuple(float(x) for x in time_profile),
        context_popularity_profile=tuple(float(x) for x in context_popularity),
        context_quality_profile=tuple(float(x) for x in context_quality),
    )


def _context_adjusted_dr_effect(df: pd.DataFrame, case: str) -> CausalResponseSummary:
    y = df["outcome"].astype(float).to_numpy()
    a = (df["treatment"].astype(float).to_numpy() > 0.5).astype(float)
    context = df["context_id"].to_numpy()
    raw = float(y[a > 0.5].mean() - y[a <= 0.5].mean()) if 0 < a.mean() < 1 else 0.0
    base_scale = max(abs(raw), float(np.mean(y)) * 0.02, 1e-4)

    if a.min() == a.max() or a.mean() < 0.02 or a.mean() > 0.98:
        return CausalResponseSummary(
            case=case,
            predicted_effect=base_scale,
            doubly_robust_effect=base_scale,
            robust_effect=max(base_scale * 0.5, 1e-4),
            doubly_robust_se=float(np.std(y) / np.sqrt(max(len(y), 1))),
            propensity_min=float(a.mean()),
            propensity_max=float(a.mean()),
            notes="insufficient treatment variation; uses conservative public-data-calibrated response scale",
        )

    work = pd.DataFrame({"y": y, "a": a, "context": context})
    global_e = float(np.clip(work["a"].mean(), 0.05, 0.95))
    global_m1 = float(work.loc[work["a"] > 0.5, "y"].mean())
    global_m0 = float(work.loc[work["a"] <= 0.5, "y"].mean())

    grouped = work.groupby("context", observed=True)
    e_by_context = grouped["a"].mean().clip(0.05, 0.95)
    m1_by_context = work[work["a"] > 0.5].groupby("context", observed=True)["y"].mean()
    m0_by_context = work[work["a"] <= 0.5].groupby("context", observed=True)["y"].mean()

    e = pd.Series(context).map(e_by_context).fillna(global_e).to_numpy(dtype=float)
    m1 = pd.Series(context).map(m1_by_context).fillna(global_m1).to_numpy(dtype=float)
    m0 = pd.Series(context).map(m0_by_context).fillna(global_m0).to_numpy(dtype=float)
    dr_score = m1 - m0 + a * (y - m1) / e - (1 - a) * (y - m0) / (1 - e)
    dr_effect = float(np.mean(dr_score))
    dr_se = float(np.std(dr_score, ddof=1) / np.sqrt(max(len(dr_score), 1)))
    robust_effect = float(max(dr_effect - 1.96 * dr_se, 1e-4))
    return CausalResponseSummary(
        case=case,
        predicted_effect=float(max(raw, 1e-4)),
        doubly_robust_effect=float(max(dr_effect, 1e-4)),
        robust_effect=robust_effect,
        doubly_robust_se=dr_se,
        propensity_min=float(e.min()),
        propensity_max=float(e.max()),
        notes="context-adjusted doubly robust score with clipped empirical propensities",
    )


def causal_response_summary(case_name: str, max_rows: int = 500_000) -> CausalResponseSummary:
    if case_name == "criteo":
        return _context_adjusted_dr_effect(load_criteo(max_rows=max_rows), "criteo")
    if case_name in {"kuairand", "kuairec"}:
        return _context_adjusted_dr_effect(load_kuairand(max_rows=max_rows), "kuairand")
    raise ValueError(f"Unknown case {case_name}")


def calibrate_kuairec(max_rows: int = 500_000) -> DataCalibration:
    """Backward-compatible alias for notebooks written before the dataset rename."""
    return calibrate_kuairand(max_rows=max_rows)


def inspect_kuairand_archive() -> pd.DataFrame:
    path = DATA_DIR / "KuaiRand" / "10439422.zip"
    if not path.exists():
        return pd.DataFrame()
    with ZipFile(path) as zf:
        rows = [{"name": info.filename, "size_mb": info.file_size / 1_000_000} for info in zf.infolist()]
    return pd.DataFrame(rows)
