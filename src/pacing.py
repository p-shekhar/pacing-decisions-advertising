from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from .forecasting import conformal_quantile
from .simulation import StreamingCase


@dataclass(frozen=True)
class Policy:
    name: str
    intensity: float
    quality_focus: float
    demand_focus: float
    load_guard: float
    budget_guard: float


@dataclass
class PolicyEvaluation:
    policy: str
    value: float
    constraints: np.ndarray
    delivery: float
    spend: float
    member_load: float


def make_policy_catalog(granularity: str = "medium") -> list[Policy]:
    if granularity == "coarse":
        intensities = [0.45, 0.75, 1.0]
        quality = [0.2, 0.7]
    elif granularity == "fine":
        intensities = [0.35, 0.5, 0.65, 0.8, 0.95, 1.1]
        quality = [0.0, 0.35, 0.7, 1.0]
    else:
        intensities = [0.4, 0.6, 0.8, 1.0]
        quality = [0.15, 0.6, 0.95]
    policies = []
    for i, intensity in enumerate(intensities):
        for j, q_focus in enumerate(quality):
            policies.append(
                Policy(
                    name=f"pace_i{i}_q{j}",
                    intensity=float(intensity),
                    quality_focus=float(q_focus),
                    demand_focus=float(1.0 - 0.5 * q_focus),
                    load_guard=float(0.8 + 0.4 * q_focus),
                    budget_guard=float(1.15 - 0.25 * intensity),
                )
            )
    return policies


def _reshape_z(z: np.ndarray) -> np.ndarray:
    if z.ndim != 1:
        z = z.reshape(-1)
    if z.size % 3 != 0:
        raise ValueError("Expected z to have three coordinates per segment.")
    return z.reshape(-1, 3)


def _segment_campaign_profile(
    case: StreamingCase,
    column: str,
    n_segments: int,
    normalize: bool = True,
) -> np.ndarray:
    preferred = case.campaign_table["preferred_segment"].to_numpy(dtype=int) % n_segments
    weights = case.campaign_table[column].to_numpy(dtype=float)
    totals = np.bincount(preferred, weights=weights, minlength=n_segments).astype(float)
    counts = np.bincount(preferred, minlength=n_segments).astype(float)
    fallback = float(np.mean(weights)) if len(weights) else 1.0
    profile = totals / np.maximum(counts, 1.0)
    profile[counts == 0] = fallback
    if normalize:
        profile = profile / max(float(np.mean(profile)), 1e-9)
    return profile


def policy_coefficients(policy: Policy, case: StreamingCase) -> tuple[np.ndarray, np.ndarray]:
    """Return value and constraint coefficients for an affine policy evaluation.

    Coordinates are inventory mass, demand-pressure mass, and quality-adjusted
    mass for each segment. Constraint order is delivery shortfall, budget
    overspend, member-experience load.
    """
    n_segments = len(case.segment_table)
    seg_pop = case.segment_table["segment_popularity"].to_numpy()
    seg_pop = seg_pop / seg_pop.mean()
    value_profile = _segment_campaign_profile(case, "value_per_incremental", n_segments)
    delivery_profile = _segment_campaign_profile(case, "delivery_target", n_segments)
    budget_profile = _segment_campaign_profile(case, "budget", n_segments)
    inv = np.zeros(n_segments)
    demand = np.zeros(n_segments)
    qual = np.zeros(n_segments)

    inv += policy.intensity * (0.16 + 0.04 * seg_pop)
    demand += policy.intensity * policy.demand_focus * 0.08
    qual += policy.intensity * policy.quality_focus * 0.22

    coeff = np.column_stack([inv, demand, qual]).reshape(-1)
    response = case.response_scale
    value_per = float(case.campaign_table["value_per_incremental"].mean())
    value_coeff = coeff * response * value_per * np.repeat(value_profile, 3)

    delivery_coeff = -coeff * np.repeat(delivery_profile, 3)
    spend_coeff = (
        coeff
        * np.repeat(budget_profile, 3)
        * (0.035 + 0.05 * policy.demand_focus)
        / max(policy.budget_guard, 1e-6)
    )
    load_coeff = coeff * case.member_scale / max(policy.load_guard, 1e-6)
    constraint_coeffs = np.vstack([delivery_coeff, spend_coeff, load_coeff])
    return value_coeff, constraint_coeffs


def policy_targets(case: StreamingCase) -> np.ndarray:
    delivery_target = float(case.campaign_table["delivery_target"].sum())
    budget = float(case.campaign_table["budget"].sum())
    max_load = float(case.campaign_table["max_member_load"].sum())
    return np.array([delivery_target, -budget, -max_load], dtype=float)


def evaluate_policy(policy: Policy, case: StreamingCase, z: np.ndarray) -> PolicyEvaluation:
    value_coeff, constraint_coeffs = policy_coefficients(policy, case)
    targets = policy_targets(case)
    delivered = float(-constraint_coeffs[0] @ z)
    spend = float(constraint_coeffs[1] @ z)
    member_load = float(constraint_coeffs[2] @ z)
    constraints = constraint_coeffs @ z + targets
    value = float(value_coeff @ z)
    return PolicyEvaluation(policy.name, value, constraints, delivered, spend, member_load)


def evaluation_frame(policies: list[Policy], case: StreamingCase, z: np.ndarray) -> pd.DataFrame:
    rows = []
    for policy in policies:
        ev = evaluate_policy(policy, case, z)
        rows.append(
            {
                "policy": ev.policy,
                "value": ev.value,
                "delivery_shortfall": ev.constraints[0],
                "budget_overspend": ev.constraints[1],
                "member_load_violation": ev.constraints[2],
                "delivery": ev.delivery,
                "spend": ev.spend,
                "member_load": ev.member_load,
                "feasible": bool(np.all(ev.constraints <= 0)),
            }
        )
    return pd.DataFrame(rows)


def nominal_dual_prices(policies: list[Policy], case: StreamingCase, z_hat: np.ndarray) -> np.ndarray:
    values = []
    constraints = []
    for policy in policies:
        ev = evaluate_policy(policy, case, z_hat)
        values.append(ev.value)
        constraints.append(ev.constraints)
    values = np.asarray(values)
    constraints = np.asarray(constraints)
    n = len(policies)
    a_ub = constraints.T
    b_ub = np.zeros(a_ub.shape[0])
    a_eq = np.ones((1, n))
    b_eq = np.array([1.0])
    bounds = [(0.0, 1.0)] * n
    result = linprog(
        c=-values,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if result.success and hasattr(result, "ineqlin"):
        marginals = np.asarray(result.ineqlin.marginals, dtype=float)
        return np.clip(-marginals, 0, None)
    return np.zeros(a_ub.shape[0], dtype=float)


def sensitivity_matrix(policies: list[Policy], case: StreamingCase, lambdas: np.ndarray) -> np.ndarray:
    rows = []
    for policy in policies:
        value_coeff, constraint_coeffs = policy_coefficients(policy, case)
        rows.append(value_coeff - lambdas @ constraint_coeffs)
    return np.vstack(rows)


def component_sensitivity_matrices(
    policies: list[Policy],
    case: StreamingCase,
    lambdas: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return all forecast-error directions used by the selector.

    The Lagrangian matrix is the sharp support-function object used in the
    geometry theorem. The value and constraint matrices are calibrated
    separately so that the implemented robust selector certifies the same
    lower-value and upper-constraint quantities that it reports.
    """
    value_rows = []
    delivery_rows = []
    budget_rows = []
    member_rows = []
    lagrangian_rows = []
    for policy in policies:
        value_coeff, constraint_coeffs = policy_coefficients(policy, case)
        value_rows.append(value_coeff)
        delivery_rows.append(constraint_coeffs[0])
        budget_rows.append(constraint_coeffs[1])
        member_rows.append(constraint_coeffs[2])
        lagrangian_rows.append(value_coeff - lambdas @ constraint_coeffs)
    return {
        "lagrangian": np.vstack(lagrangian_rows),
        "value": np.vstack(value_rows),
        "delivery": np.vstack(delivery_rows),
        "budget": np.vstack(budget_rows),
        "member": np.vstack(member_rows),
    }


def decision_scores(w: np.ndarray, errors: np.ndarray) -> np.ndarray:
    return np.max(np.abs(errors @ w.T), axis=1)


def generic_scores(errors: np.ndarray) -> np.ndarray:
    return np.linalg.norm(errors, axis=1)


def _normal_radius_multiplier(alpha: float) -> float:
    alpha = min(max(float(alpha), 1e-9), 1 - 1e-9)
    return float(NormalDist().inv_cdf(1 - alpha / 2))


def _positive_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        return None
    return value


def policy_radii(
    policy: Policy,
    case: StreamingCase,
    causal_radius_scale: float = 1.0,
    alpha_tau: float = 0.1,
    alpha_m: float = 0.1,
    tau_se: float | None = None,
    member_se: float | None = None,
) -> tuple[float, np.ndarray]:
    value_coeff, constraint_coeffs = policy_coefficients(policy, case)
    z_tau = _normal_radius_multiplier(alpha_tau)
    z_member = _normal_radius_multiplier(alpha_m)

    response_scale = max(abs(float(case.response_scale)), 1e-6)
    member_scale = max(abs(float(case.member_scale)), 1e-6)
    tau_std = _positive_or_none(tau_se)
    member_std = _positive_or_none(member_se)
    if tau_std is None:
        tau_std = 0.15 * max(response_scale, 1e-4)
    if member_std is None:
        member_std = 0.10 * max(member_scale, 1e-4)

    response_radius = causal_radius_scale * z_tau * tau_std
    value_radius = float(response_radius * np.linalg.norm(value_coeff / response_scale, ord=1))
    member_radius = float(
        causal_radius_scale
        * z_member
        * member_std
        * np.linalg.norm(constraint_coeffs[2] / member_scale, ord=1)
    )
    constraint_radius = np.array(
        [
            0.0,
            0.0,
            member_radius,
        ]
    )
    return value_radius, constraint_radius


def robust_select(
    policies: list[Policy],
    case: StreamingCase,
    z_hat: np.ndarray,
    radii: dict[str, float] | float,
    causal_radius_scale: float = 1.0,
    alpha_tau: float = 0.1,
    alpha_m: float = 0.1,
    planning_tolerance: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if isinstance(radii, dict):
        q_value = float(radii.get("q_value", radii.get("q_decision", 0.0)))
        q_constraints = np.array(
            [
                float(radii.get("q_delivery", radii.get("q_decision", 0.0))),
                float(radii.get("q_budget", radii.get("q_decision", 0.0))),
                float(radii.get("q_member", radii.get("q_decision", 0.0))),
            ]
        )
        q_lagrangian = float(radii.get("q_decision", max(q_value, float(q_constraints.max()))))
        q_generic = float(radii.get("q_generic", np.nan))
        tau_se = _positive_or_none(radii.get("tau_se"))
        member_se = _positive_or_none(radii.get("member_se"))
        effect_radius_source = str(radii.get("effect_radius_source", ""))
    else:
        q_value = float(radii)
        q_constraints = np.repeat(float(radii), 3)
        q_lagrangian = float(radii)
        q_generic = float(radii)
        tau_se = None
        member_se = None
        effect_radius_source = ""
    rows = []
    for policy in policies:
        ev = evaluate_policy(policy, case, z_hat)
        rho_v, rho_g = policy_radii(
            policy,
            case,
            causal_radius_scale=causal_radius_scale,
            alpha_tau=alpha_tau,
            alpha_m=alpha_m,
            tau_se=tau_se,
            member_se=member_se,
        )
        robust_value = ev.value - q_value - rho_v
        robust_constraints = ev.constraints + q_constraints + rho_g
        rows.append(
            {
                "policy": policy.name,
                "nominal_value": ev.value,
                "robust_value": robust_value,
                "q_radius": q_lagrangian,
                "q_value": q_value,
                "q_delivery": q_constraints[0],
                "q_budget": q_constraints[1],
                "q_member": q_constraints[2],
                "q_generic": q_generic,
                "rho_value": rho_v,
                "rho_delivery": rho_g[0],
                "rho_budget": rho_g[1],
                "rho_member": rho_g[2],
                "tau_se": tau_se if tau_se is not None else np.nan,
                "member_se": member_se if member_se is not None else np.nan,
                "effect_radius_source": effect_radius_source,
                "alpha_tau": alpha_tau,
                "alpha_m": alpha_m,
                "delivery_certified": robust_constraints[0] <= 0,
                "budget_certified": robust_constraints[1] <= 0,
                "member_certified": robust_constraints[2] <= 0,
                "all_constraints_certified": bool(np.all(robust_constraints <= 0)),
                "robust_delivery_shortfall": robust_constraints[0],
                "robust_budget_overspend": robust_constraints[1],
                "robust_member_violation": robust_constraints[2],
            }
        )
    df = pd.DataFrame(rows)
    feasible = df[df["all_constraints_certified"]].copy()
    if feasible.empty:
        df["selected"] = False
        best_value = df["robust_value"].max()
        df["shortlist"] = df["robust_value"] >= best_value - abs(best_value) * planning_tolerance
        return df.sort_values("robust_value", ascending=False), feasible
    best_value = feasible["robust_value"].max()
    feasible["selected"] = feasible["robust_value"] == best_value
    df = df.merge(feasible[["policy", "selected"]], on="policy", how="left")
    df["selected"] = df["selected"].fillna(False)
    df["shortlist"] = df["robust_value"] >= best_value - abs(best_value) * planning_tolerance
    return df.sort_values("robust_value", ascending=False), feasible.sort_values("robust_value", ascending=False)


def calibrate_radii(
    policies: list[Policy],
    case: StreamingCase,
    z: np.ndarray,
    z_hat: np.ndarray,
    cal_idx: np.ndarray,
    alpha: float = 0.1,
    joint_components: bool = True,
) -> dict[str, float]:
    z_hat_cal = z_hat[cal_idx]
    z_cal = z[cal_idx]
    errors = z_cal - z_hat_cal
    lambdas_by_block = []
    lagrangian_scores = []
    for row, err in zip(z_hat_cal, errors):
        lambdas_b = nominal_dual_prices(policies, case, row)
        lambdas_by_block.append(lambdas_b)
        components_b = component_sensitivity_matrices(policies, case, lambdas_b)
        lagrangian_scores.append(decision_scores(components_b["lagrangian"], err[None, :])[0])
    lambdas = np.mean(np.vstack(lambdas_by_block), axis=0)
    lambda_std = np.std(np.vstack(lambdas_by_block), axis=0)
    components = component_sensitivity_matrices(policies, case, lambdas)
    component_alpha = alpha / 4 if joint_components else alpha
    return {
        "q_decision": conformal_quantile(np.asarray(lagrangian_scores), alpha=alpha),
        "q_value": conformal_quantile(decision_scores(components["value"], errors), alpha=component_alpha),
        "q_delivery": conformal_quantile(decision_scores(components["delivery"], errors), alpha=component_alpha),
        "q_budget": conformal_quantile(decision_scores(components["budget"], errors), alpha=component_alpha),
        "q_member": conformal_quantile(decision_scores(components["member"], errors), alpha=component_alpha),
        "q_generic": conformal_quantile(generic_scores(errors), alpha=alpha),
        "alpha": float(alpha),
        "component_alpha": float(component_alpha),
        "dual_mode": "lp_duals_block_specific_for_lagrangian_mean_reported_for_components",
        "n_calibration_blocks": float(len(cal_idx)),
        "lambda_delivery": float(lambdas[0]),
        "lambda_budget": float(lambdas[1]),
        "lambda_member": float(lambdas[2]),
        "lambda_delivery_sd": float(lambda_std[0]),
        "lambda_budget_sd": float(lambda_std[1]),
        "lambda_member_sd": float(lambda_std[2]),
    }
