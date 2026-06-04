from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .data import calibrate_criteo, calibrate_kuairand, causal_response_summary
from .forecasting import fit_all_forecasters, forecast_summary_table, split_blocks
from .pacing import (
    calibrate_radii,
    evaluation_frame,
    make_policy_catalog,
    nominal_dual_prices,
    robust_select,
)
from .simulation import StreamingCase, generate_streaming_case


@dataclass
class CaseBundle:
    case: StreamingCase
    splits: dict[str, np.ndarray]
    policies: list
    forecasts: dict


def build_case_bundle(
    case_name: str,
    max_rows: int = 500_000,
    seed: int = 13,
    n_blocks: int = 120,
    model_names: list[str] | tuple[str, ...] | set[str] | None = None,
) -> CaseBundle:
    if case_name == "criteo":
        calibration = calibrate_criteo(max_rows=max_rows)
    elif case_name in {"kuairand", "kuairec"}:
        calibration = calibrate_kuairand(max_rows=max_rows)
    else:
        raise ValueError(f"Unknown case {case_name}")
    case = generate_streaming_case(calibration, n_blocks=n_blocks, seed=seed)
    splits = split_blocks(case.z_blocks.shape[0])
    policies = make_policy_catalog("medium")
    forecasts = fit_all_forecasters(
        case.z_blocks,
        splits["train"],
        history=case.history_features,
        model_names=model_names,
    )
    return CaseBundle(case, splits, policies, forecasts)


def _planning_vectors(bundle: CaseBundle, model: str) -> tuple[np.ndarray, np.ndarray]:
    test_idx = bundle.splits["test"]
    z_true = bundle.case.z_blocks[test_idx].mean(axis=0)
    z_hat = bundle.forecasts[model].predictions[test_idx].mean(axis=0)
    return z_true, z_hat


def _attach_effect_radii(
    radii: dict[str, float],
    bundle: CaseBundle,
    response: object,
) -> dict[str, float]:
    """Attach empirical causal-response and member-experience uncertainty scales."""
    out = dict(radii)
    cal_idx = bundle.splits["cal"]
    z_cal = bundle.case.z_blocks[cal_idx]
    quality_mass = z_cal[:, 2::3].sum(axis=1)
    member_relative_se = float(
        np.std(quality_mass, ddof=1)
        / np.sqrt(max(len(quality_mass), 1))
        / max(abs(float(np.mean(quality_mass))), 1e-9)
    )
    tau_se = float(getattr(response, "doubly_robust_se", np.nan))
    tau_point = float(getattr(response, "doubly_robust_effect", bundle.case.response_scale))
    if not np.isfinite(tau_se) or tau_se <= 0:
        tau_se = 0.10 * max(abs(tau_point), abs(bundle.case.response_scale), 1e-4)
    out["tau_se"] = tau_se
    out["member_se"] = max(member_relative_se * max(bundle.case.member_scale, 0.03), 1e-6)
    out["effect_radius_source"] = "context_adjusted_dr_se_and_calibration_member_variability"
    return out


def _realized_policy_metrics(
    policies: list,
    case: StreamingCase,
    z_blocks: np.ndarray,
    selected_policy: str,
) -> dict[str, float | bool]:
    rows = []
    for z in z_blocks:
        frame = evaluation_frame(policies, case, z)
        rows.append(frame[frame["policy"] == selected_policy].iloc[0])
    realized = pd.DataFrame(rows)
    any_block_violation = (
        (realized["delivery_shortfall"] > 0)
        | (realized["budget_overspend"] > 0)
        | (realized["member_load_violation"] > 0)
    )
    return {
        "true_value": float(realized["value"].mean()),
        "delivery_violation": bool((realized["delivery_shortfall"].mean()) > 0),
        "budget_violation": bool((realized["budget_overspend"].mean()) > 0),
        "member_violation": bool((realized["member_load_violation"].mean()) > 0),
        "any_violation": bool(any_block_violation.any()),
        "delivery_violation_rate": float((realized["delivery_shortfall"] > 0).mean()),
        "budget_violation_rate": float((realized["budget_overspend"] > 0).mean()),
        "member_violation_rate": float((realized["member_load_violation"] > 0).mean()),
        "any_violation_rate": float(any_block_violation.mean()),
        "mean_delivery_shortfall": float(realized["delivery_shortfall"].mean()),
        "mean_budget_overspend": float(realized["budget_overspend"].mean()),
        "mean_member_violation": float(realized["member_load_violation"].mean()),
    }


def _empty_realized_metrics() -> dict[str, float | bool]:
    return {
        "true_value": np.nan,
        "delivery_violation": np.nan,
        "budget_violation": np.nan,
        "member_violation": np.nan,
        "any_violation": np.nan,
        "delivery_violation_rate": np.nan,
        "budget_violation_rate": np.nan,
        "member_violation_rate": np.nan,
        "any_violation_rate": np.nan,
        "mean_delivery_shortfall": np.nan,
        "mean_budget_overspend": np.nan,
        "mean_member_violation": np.nan,
    }


def _zero_radii_like(radii: dict[str, float]) -> dict[str, float]:
    out = dict(radii)
    for key in ["q_decision", "q_value", "q_delivery", "q_budget", "q_member", "q_generic"]:
        out[key] = 0.0
    return out


def _generic_radii_like(radii: dict[str, float]) -> dict[str, float]:
    out = dict(radii)
    q = float(radii["q_generic"])
    for key in ["q_decision", "q_value", "q_delivery", "q_budget", "q_member"]:
        out[key] = q
    return out


def compare_pacing_methods(
    case_name: str,
    model: str = "numpy_mlp",
    max_rows: int = 500_000,
    seed: int = 13,
    alpha: float = 0.1,
    alpha_tau: float = 0.1,
    alpha_m: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bundle = build_case_bundle(case_name, max_rows=max_rows, seed=seed)
    z_true, z_hat = _planning_vectors(bundle, model)
    z_test_blocks = bundle.case.z_blocks[bundle.splits["test"]]
    response = causal_response_summary(case_name, max_rows=max_rows)
    radii = _attach_effect_radii(
        calibrate_radii(
            bundle.policies,
            bundle.case,
            bundle.case.z_blocks,
            bundle.forecasts[model].predictions,
            bundle.splits["cal"],
            alpha=alpha,
        ),
        bundle,
        response,
    )
    methods = {
        "point_forecast": (_zero_radii_like(radii), 0.0),
        "generic_residual_conformal": (_generic_radii_like(radii), 1.0),
        "decision_calibrated": (radii, 1.0),
    }
    rows = []
    score_frames = []
    true_eval = evaluation_frame(bundle.policies, bundle.case, z_true)
    best_true_feasible = true_eval[true_eval["feasible"]]["value"].max()
    if not np.isfinite(best_true_feasible):
        best_true_feasible = true_eval["value"].max()
    for method, (method_radii, causal_radius_scale) in methods.items():
        scores, _ = robust_select(
            bundle.policies,
            bundle.case,
            z_hat,
            method_radii,
            causal_radius_scale=causal_radius_scale,
            alpha_tau=alpha_tau,
            alpha_m=alpha_m,
        )
        selected = scores[scores["selected"]]
        has_certified = not selected.empty
        representative_policy = scores.iloc[0]["policy"]
        if selected.empty:
            selected_policy = "unresolved"
            realized = _empty_realized_metrics()
        else:
            selected_policy = selected.iloc[0]["policy"]
            realized = _realized_policy_metrics(bundle.policies, bundle.case, z_test_blocks, selected_policy)
        row = {
                "case": case_name,
                "model": model,
                "method": method,
                "selected_policy": selected_policy,
                "representative_policy_if_unresolved": representative_policy,
                "has_certified_policy": has_certified,
                "q_radius": method_radii["q_decision"],
                "q_value": method_radii["q_value"],
                "q_delivery": method_radii["q_delivery"],
                "q_budget": method_radii["q_budget"],
                "q_member": method_radii["q_member"],
                "regret_to_best_true_feasible": np.nan
                if not has_certified
                else best_true_feasible - float(realized["true_value"]),
                "shortlist_size": int(scores["shortlist"].sum()),
                "q_generic": method_radii["q_generic"],
                "lambda_delivery": method_radii["lambda_delivery"],
                "lambda_budget": method_radii["lambda_budget"],
                "lambda_member": method_radii["lambda_member"],
                "causal_radius_scale": causal_radius_scale,
                "alpha": alpha,
                "alpha_tau": alpha_tau,
                "alpha_m": alpha_m,
                "component_alpha": method_radii.get("component_alpha", np.nan),
                "dual_mode": method_radii.get("dual_mode", ""),
                "tau_se": method_radii.get("tau_se", np.nan),
                "member_se": method_radii.get("member_se", np.nan),
                "effect_radius_source": method_radii.get("effect_radius_source", ""),
            }
        row.update(realized)
        rows.append(row)
        sf = scores.copy()
        sf["case"] = case_name
        sf["method"] = method
        score_frames.append(sf)
    return pd.DataFrame(rows), pd.concat(score_frames, ignore_index=True)


def geometry_vs_robustness_ablation(
    case_name: str,
    model: str = "numpy_mlp",
    max_rows: int = 500_000,
    seed: int = 13,
    alpha: float = 0.1,
    alpha_tau: float = 0.1,
    alpha_m: float = 0.1,
) -> pd.DataFrame:
    """Decompose gains from robust selection and support-function geometry.

    The generic and decision-calibrated rows use the same robust optimizer.
    Their difference isolates the radius geometry. The final audit row takes
    the nominal point-forecast policy and evaluates whether it would pass the
    decision-calibrated robust certificate, isolating the role of robust
    optimization itself.
    """
    bundle = build_case_bundle(case_name, max_rows=max_rows, seed=seed, model_names=[model])
    z_true, z_hat = _planning_vectors(bundle, model)
    z_test_blocks = bundle.case.z_blocks[bundle.splits["test"]]
    response = causal_response_summary(case_name, max_rows=max_rows)
    radii = _attach_effect_radii(
        calibrate_radii(
            bundle.policies,
            bundle.case,
            bundle.case.z_blocks,
            bundle.forecasts[model].predictions,
            bundle.splits["cal"],
            alpha=alpha,
        ),
        bundle,
        response,
    )
    variants = [
        (
            "point_forecast_nominal",
            "none",
            "nominal selection with zero uncertainty radii",
            _zero_radii_like(radii),
            0.0,
        ),
        (
            "generic_robust_same_optimizer",
            "generic_residual",
            "robust selector with unweighted residual radius",
            _generic_radii_like(radii),
            1.0,
        ),
        (
            "support_geometry_robust",
            "support_function",
            "same robust selector with support-function component radii",
            radii,
            1.0,
        ),
    ]
    rows = []
    score_by_variant: dict[str, pd.DataFrame] = {}
    point_policy = "unresolved"
    for method, radius_geometry, role, method_radii, causal_radius_scale in variants:
        scores, _ = robust_select(
            bundle.policies,
            bundle.case,
            z_hat,
            method_radii,
            causal_radius_scale=causal_radius_scale,
            alpha_tau=alpha_tau,
            alpha_m=alpha_m,
        )
        score_by_variant[method] = scores
        selected = scores[scores["selected"]]
        has_certified = not selected.empty
        representative = selected.iloc[0] if has_certified else scores.iloc[0]
        selected_policy = representative["policy"] if has_certified else "unresolved"
        if method == "point_forecast_nominal" and has_certified:
            point_policy = str(selected_policy)
        realized = (
            _realized_policy_metrics(bundle.policies, bundle.case, z_test_blocks, str(selected_policy))
            if has_certified
            else _empty_realized_metrics()
        )
        row = {
            "case": case_name,
            "model": model,
            "ablation_row": method,
            "radius_geometry": radius_geometry,
            "comparison_role": role,
            "selected_policy": selected_policy,
            "audited_policy": selected_policy,
            "has_certified_policy": has_certified,
            "same_robust_optimizer_as_decision": method != "point_forecast_nominal",
            "q_radius": method_radii["q_decision"],
            "q_generic": method_radii["q_generic"],
            "q_value": method_radii["q_value"],
            "q_delivery": method_radii["q_delivery"],
            "q_budget": method_radii["q_budget"],
            "q_member": method_radii["q_member"],
            "robust_value": float(representative["robust_value"]),
            "robust_delivery_shortfall": float(representative["robust_delivery_shortfall"]),
            "robust_budget_overspend": float(representative["robust_budget_overspend"]),
            "robust_member_violation": float(representative["robust_member_violation"]),
            "shortlist_size": int(scores["shortlist"].sum()),
            "alpha": alpha,
            "alpha_tau": alpha_tau,
            "alpha_m": alpha_m,
            "component_alpha": method_radii.get("component_alpha", np.nan),
            "dual_mode": method_radii.get("dual_mode", ""),
            "tau_se": method_radii.get("tau_se", np.nan),
            "member_se": method_radii.get("member_se", np.nan),
            "effect_radius_source": method_radii.get("effect_radius_source", ""),
        }
        row.update(realized)
        rows.append(row)

    if point_policy != "unresolved":
        decision_scores = score_by_variant["support_geometry_robust"]
        audited = decision_scores[decision_scores["policy"] == point_policy].iloc[0]
        audit_realized = _realized_policy_metrics(bundle.policies, bundle.case, z_test_blocks, point_policy)
        audit_row = {
            "case": case_name,
            "model": model,
            "ablation_row": "point_policy_under_support_certificate",
            "radius_geometry": "support_function",
            "comparison_role": "nominally selected policy audited under decision-calibrated robust radii",
            "selected_policy": "audit_only",
            "audited_policy": point_policy,
            "has_certified_policy": bool(audited["all_constraints_certified"]),
            "same_robust_optimizer_as_decision": False,
            "q_radius": radii["q_decision"],
            "q_generic": radii["q_generic"],
            "q_value": radii["q_value"],
            "q_delivery": radii["q_delivery"],
            "q_budget": radii["q_budget"],
            "q_member": radii["q_member"],
            "robust_value": float(audited["robust_value"]),
            "robust_delivery_shortfall": float(audited["robust_delivery_shortfall"]),
            "robust_budget_overspend": float(audited["robust_budget_overspend"]),
            "robust_member_violation": float(audited["robust_member_violation"]),
            "shortlist_size": int(decision_scores["shortlist"].sum()),
            "alpha": alpha,
            "alpha_tau": alpha_tau,
            "alpha_m": alpha_m,
            "component_alpha": radii.get("component_alpha", np.nan),
            "dual_mode": radii.get("dual_mode", ""),
            "tau_se": radii.get("tau_se", np.nan),
            "member_se": radii.get("member_se", np.nan),
            "effect_radius_source": radii.get("effect_radius_source", ""),
        }
        audit_row.update(audit_realized)
        rows.append(audit_row)

    return pd.DataFrame(rows)


def compare_forecasters(
    case_name: str,
    max_rows: int = 500_000,
    seed: int = 13,
    alpha: float = 0.1,
    alpha_tau: float = 0.1,
    alpha_m: float = 0.1,
) -> pd.DataFrame:
    bundle = build_case_bundle(case_name, max_rows=max_rows, seed=seed)
    z_test_blocks = bundle.case.z_blocks[bundle.splits["test"]]
    response = causal_response_summary(case_name, max_rows=max_rows)
    rows = []
    for model, result in bundle.forecasts.items():
        z_true, z_hat = _planning_vectors(bundle, model)
        radii = _attach_effect_radii(
            calibrate_radii(
                bundle.policies,
                bundle.case,
                bundle.case.z_blocks,
                result.predictions,
                bundle.splits["cal"],
                alpha=alpha,
            ),
            bundle,
            response,
        )
        scores, _ = robust_select(
            bundle.policies,
            bundle.case,
            z_hat,
            radii,
            alpha_tau=alpha_tau,
            alpha_m=alpha_m,
        )
        selected = scores[scores["selected"]]
        has_certified = not selected.empty
        if selected.empty:
            selected_policy = "unresolved"
            representative = scores.iloc[0]["policy"]
            robust_value = np.nan
            realized = _empty_realized_metrics()
        else:
            selected_policy = selected.iloc[0]["policy"]
            representative = selected_policy
            robust_value = float(selected.iloc[0]["robust_value"])
            realized = _realized_policy_metrics(bundle.policies, bundle.case, z_test_blocks, selected_policy)
        row = {
                "case": case_name,
                "model": model,
                "test_mse": float(
                    np.mean((bundle.case.z_blocks[bundle.splits["test"]] - result.predictions[bundle.splits["test"]]) ** 2)
                ),
                "test_mae": float(
                    np.mean(np.abs(bundle.case.z_blocks[bundle.splits["test"]] - result.predictions[bundle.splits["test"]]))
                ),
                "band_radius": result.band_radius,
                "band_coverage": float(
                    np.mean(
                        np.all(
                            (bundle.case.z_blocks[bundle.splits["test"]] >= result.lower[bundle.splits["test"]])
                            & (bundle.case.z_blocks[bundle.splits["test"]] <= result.upper[bundle.splits["test"]]),
                            axis=1,
                        )
                    )
                )
                if result.lower is not None and result.upper is not None
                else np.nan,
                "mean_band_width": float(
                    np.mean(result.upper[bundle.splits["test"]] - result.lower[bundle.splits["test"]])
                )
                if result.lower is not None and result.upper is not None
                else np.nan,
                "q_decision": radii["q_decision"],
                "q_value": radii["q_value"],
                "q_delivery": radii["q_delivery"],
                "q_budget": radii["q_budget"],
                "q_member": radii["q_member"],
                "q_generic": radii["q_generic"],
                "component_alpha": radii.get("component_alpha", np.nan),
                "dual_mode": radii.get("dual_mode", ""),
                "n_calibration_blocks": radii.get("n_calibration_blocks", np.nan),
                "tau_se": radii.get("tau_se", np.nan),
                "member_se": radii.get("member_se", np.nan),
                "selected_policy": selected_policy,
                "representative_policy_if_unresolved": representative,
                "has_certified_policy": has_certified,
                "robust_value": robust_value,
            }
        row.update(realized)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("q_decision")


def compare_causal_response_modes(
    case_name: str,
    model: str = "numpy_mlp",
    max_rows: int = 500_000,
    seed: int = 13,
    alpha: float = 0.1,
    alpha_tau: float = 0.1,
    alpha_m: float = 0.1,
) -> pd.DataFrame:
    bundle = build_case_bundle(case_name, max_rows=max_rows, seed=seed)
    _, z_hat = _planning_vectors(bundle, model)
    z_true, _ = _planning_vectors(bundle, model)
    z_test_blocks = bundle.case.z_blocks[bundle.splits["test"]]
    response = causal_response_summary(case_name, max_rows=max_rows)
    dr_scale = max(
        0.05,
        min(2.0, 1.96 * response.doubly_robust_se / max(0.25 * abs(response.doubly_robust_effect), 1e-6)),
    )
    modes = {
        "predicted_response": {
            "response_scale": response.predicted_effect,
            "causal_radius_scale": 0.0,
        },
        "doubly_robust_response": {
            "response_scale": response.doubly_robust_effect,
            "causal_radius_scale": dr_scale,
        },
        "robust_causal_response": {
            "response_scale": response.robust_effect,
            "causal_radius_scale": 1.0,
        },
    }
    rows = []
    for mode, spec in modes.items():
        case_mode = replace(bundle.case, response_scale=max(float(spec["response_scale"]), 1e-4))
        radii = _attach_effect_radii(
            calibrate_radii(
                bundle.policies,
                case_mode,
                bundle.case.z_blocks,
                bundle.forecasts[model].predictions,
                bundle.splits["cal"],
                alpha=alpha,
            ),
            bundle,
            response,
        )
        scores, _ = robust_select(
            bundle.policies,
            case_mode,
            z_hat,
            radii,
            causal_radius_scale=float(spec["causal_radius_scale"]),
            alpha_tau=alpha_tau,
            alpha_m=alpha_m,
        )
        selected = scores[scores["selected"]]
        has_certified = not selected.empty
        if selected.empty:
            selected_policy = "unresolved"
            representative = scores.iloc[0]["policy"]
            realized = _empty_realized_metrics()
            robust_value = np.nan
        else:
            selected_policy = selected.iloc[0]["policy"]
            representative = selected_policy
            realized = _realized_policy_metrics(bundle.policies, bundle.case, z_test_blocks, selected_policy)
            robust_value = selected.iloc[0]["robust_value"]
        row = {
                "case": case_name,
                "mode": mode,
                "response_estimate": spec["response_scale"],
                "doubly_robust_se": response.doubly_robust_se,
                "causal_radius_scale": spec["causal_radius_scale"],
                "alpha_tau": alpha_tau,
                "alpha_m": alpha_m,
                "tau_se": radii.get("tau_se", np.nan),
                "member_se": radii.get("member_se", np.nan),
                "response_notes": response.notes,
                "selected_policy": selected_policy,
                "representative_policy_if_unresolved": representative,
                "has_certified_policy": has_certified,
                "robust_value": robust_value,
            }
        row.update(realized)
        rows.append(row)
    return pd.DataFrame(rows)


def causal_response_misspecification_ablation(
    case_name: str = "criteo",
    model: str = "numpy_mlp",
    max_rows: int = 500_000,
    seed: int = 13,
    alpha: float = 0.1,
    alpha_tau: float = 0.1,
    alpha_m: float = 0.1,
) -> pd.DataFrame:
    """Stress-test selector behavior when the response model is misspecified.

    The selector is run with a perturbed response scale, while realized value and
    feasibility are evaluated against the unperturbed public-data-calibrated
    streaming case. This implements the paper's causal-response
    misspecification diagnostic.
    """
    bundle = build_case_bundle(case_name, max_rows=max_rows, seed=seed)
    z_true, z_hat = _planning_vectors(bundle, model)
    z_test_blocks = bundle.case.z_blocks[bundle.splits["test"]]
    response = causal_response_summary(case_name, max_rows=max_rows)
    true_eval = evaluation_frame(bundle.policies, bundle.case, z_true)
    best_true_feasible = true_eval[true_eval["feasible"]]["value"].max()
    if not np.isfinite(best_true_feasible):
        best_true_feasible = true_eval["value"].max()

    rows = []
    for response_multiplier in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        case_mode = replace(
            bundle.case,
            response_scale=max(bundle.case.response_scale * response_multiplier, 1e-4),
        )
        radii = _attach_effect_radii(
            calibrate_radii(
                bundle.policies,
                case_mode,
                bundle.case.z_blocks,
                bundle.forecasts[model].predictions,
                bundle.splits["cal"],
                alpha=alpha,
            ),
            bundle,
            response,
        )
        for radius_label, causal_radius_scale in [
            ("no_causal_radius", 0.0),
            ("robust_causal_radius", 1.0),
        ]:
            scores, _ = robust_select(
                bundle.policies,
                case_mode,
                z_hat,
                radii,
                causal_radius_scale=causal_radius_scale,
                alpha_tau=alpha_tau,
                alpha_m=alpha_m,
            )
            selected = scores[scores["selected"]]
            if selected.empty:
                selected_policy = "unresolved"
                representative = scores.iloc[0]["policy"]
                realized = _empty_realized_metrics()
            else:
                selected_policy = selected.iloc[0]["policy"]
                representative = selected_policy
                realized = _realized_policy_metrics(bundle.policies, bundle.case, z_test_blocks, selected_policy)
            row = {
                    "case": case_name,
                    "model": model,
                    "response_multiplier": response_multiplier,
                    "radius_mode": radius_label,
                    "assumed_response_scale": case_mode.response_scale,
                    "true_response_scale": bundle.case.response_scale,
                    "alpha_tau": alpha_tau,
                    "alpha_m": alpha_m,
                    "tau_se": radii.get("tau_se", np.nan),
                    "member_se": radii.get("member_se", np.nan),
                    "selected_policy": selected_policy,
                    "representative_policy_if_unresolved": representative,
                    "has_certified_policy": not selected.empty,
                    "q_value": radii["q_value"],
                    "q_delivery": radii["q_delivery"],
                    "q_budget": radii["q_budget"],
                    "q_member": radii["q_member"],
                    "regret_to_best_true_feasible": np.nan
                    if selected.empty
                    else best_true_feasible - float(realized["true_value"]),
                }
            row.update(realized)
            rows.append(row)
    return pd.DataFrame(rows)


def catalog_granularity_ablation(
    case_name: str = "criteo",
    model: str = "numpy_mlp",
    max_rows: int = 500_000,
    seed: int = 13,
    alpha_tau: float = 0.1,
    alpha_m: float = 0.1,
) -> pd.DataFrame:
    bundle = build_case_bundle(case_name, max_rows=max_rows, seed=seed)
    z_true, z_hat = _planning_vectors(bundle, model)
    z_test_blocks = bundle.case.z_blocks[bundle.splits["test"]]
    response = causal_response_summary(case_name, max_rows=max_rows)
    rows = []
    for granularity in ["coarse", "medium", "fine"]:
        policies = make_policy_catalog(granularity)
        radii = _attach_effect_radii(
            calibrate_radii(
                policies,
                bundle.case,
                bundle.case.z_blocks,
                bundle.forecasts[model].predictions,
                bundle.splits["cal"],
            ),
            bundle,
            response,
        )
        scores, _ = robust_select(
            policies,
            bundle.case,
            z_hat,
            radii,
            alpha_tau=alpha_tau,
            alpha_m=alpha_m,
        )
        selected = scores[scores["selected"]]
        has_certified = not selected.empty
        if selected.empty:
            selected_policy = "unresolved"
            realized = _empty_realized_metrics()
        else:
            selected_policy = selected.iloc[0]["policy"]
            realized = _realized_policy_metrics(policies, bundle.case, z_test_blocks, selected_policy)
        row = {
                "case": case_name,
                "granularity": granularity,
                "catalog_size": len(policies),
                "q_decision": radii["q_decision"],
                "q_value": radii["q_value"],
                "q_delivery": radii["q_delivery"],
                "q_budget": radii["q_budget"],
                "q_member": radii["q_member"],
                "component_alpha": radii.get("component_alpha", np.nan),
                "dual_mode": radii.get("dual_mode", ""),
                "tau_se": radii.get("tau_se", np.nan),
                "member_se": radii.get("member_se", np.nan),
                "effect_radius_source": radii.get("effect_radius_source", ""),
                "selected_policy": selected_policy,
                "has_certified_policy": has_certified,
            }
        row.update(realized)
        rows.append(row)
    return pd.DataFrame(rows)


def budget_member_pressure_ablation(
    case_name: str = "kuairand",
    model: str = "numpy_mlp",
    max_rows: int = 500_000,
    seed: int = 13,
    alpha_tau: float = 0.1,
    alpha_m: float = 0.1,
) -> pd.DataFrame:
    bundle = build_case_bundle(case_name, max_rows=max_rows, seed=seed)
    response = causal_response_summary(case_name, max_rows=max_rows)
    rows = []
    base_budget = bundle.case.campaign_table["budget"].copy()
    base_load = bundle.case.campaign_table["max_member_load"].copy()
    for budget_mult in [0.7, 1.0, 1.4]:
        for load_mult in [0.75, 1.0, 1.25]:
            case = bundle.case
            case.campaign_table["budget"] = base_budget * budget_mult
            case.campaign_table["max_member_load"] = base_load * load_mult
            z_true, z_hat = _planning_vectors(bundle, model)
            radii = _attach_effect_radii(
                calibrate_radii(
                    bundle.policies,
                    case,
                    case.z_blocks,
                    bundle.forecasts[model].predictions,
                    bundle.splits["cal"],
                ),
                bundle,
                response,
            )
            lambdas = nominal_dual_prices(bundle.policies, case, z_hat)
            scores, _ = robust_select(
                bundle.policies,
                case,
                z_hat,
                radii,
                alpha_tau=alpha_tau,
                alpha_m=alpha_m,
            )
            selected = scores[scores["selected"]]
            has_certified = not selected.empty
            representative = scores.iloc[0]
            z_test_blocks = case.z_blocks[bundle.splits["test"]]
            if selected.empty:
                selected_policy = "unresolved"
                realized = _empty_realized_metrics()
            else:
                selected_policy = selected.iloc[0]["policy"]
                realized = _realized_policy_metrics(bundle.policies, case, z_test_blocks, selected_policy)
            row = {
                    "case": case_name,
                    "budget_multiplier": budget_mult,
                    "member_load_multiplier": load_mult,
                    "lambda_delivery": lambdas[0],
                    "lambda_budget": lambdas[1],
                    "lambda_member": lambdas[2],
                    "q_decision": radii["q_decision"],
                    "q_value": radii["q_value"],
                    "q_delivery": radii["q_delivery"],
                    "q_budget": radii["q_budget"],
                    "q_member": radii["q_member"],
                    "component_alpha": radii.get("component_alpha", np.nan),
                    "dual_mode": radii.get("dual_mode", ""),
                    "tau_se": radii.get("tau_se", np.nan),
                    "member_se": radii.get("member_se", np.nan),
                    "effect_radius_source": radii.get("effect_radius_source", ""),
                    "selected_policy": selected_policy,
                    "representative_policy_if_unresolved": representative["policy"],
                    "has_certified_policy": has_certified,
                    "representative_robust_value": representative["robust_value"],
                    "representative_robust_delivery_shortfall": representative["robust_delivery_shortfall"],
                    "representative_robust_budget_overspend": representative["robust_budget_overspend"],
                    "representative_robust_member_violation": representative["robust_member_violation"],
                    "min_robust_delivery_shortfall": scores["robust_delivery_shortfall"].min(),
                    "min_robust_budget_overspend": scores["robust_budget_overspend"].min(),
                    "min_robust_member_violation": scores["robust_member_violation"].min(),
                }
            row.update(realized)
            rows.append(row)
    case.campaign_table["budget"] = base_budget
    case.campaign_table["max_member_load"] = base_load
    return pd.DataFrame(rows)


def case_forecast_diagnostics(case_name: str, max_rows: int = 500_000, seed: int = 13) -> pd.DataFrame:
    bundle = build_case_bundle(case_name, max_rows=max_rows, seed=seed)
    return forecast_summary_table(bundle.forecasts, bundle.case.z_blocks, bundle.splits["test"])


def dual_vs_unweighted_ablation(
    case_name: str = "criteo",
    max_rows: int = 500_000,
    seed: int = 13,
    alpha: float = 0.1,
) -> pd.DataFrame:
    bundle = build_case_bundle(case_name, max_rows=max_rows, seed=seed)
    rows = []
    for model, result in bundle.forecasts.items():
        radii = calibrate_radii(
            bundle.policies,
            bundle.case,
            bundle.case.z_blocks,
            result.predictions,
            bundle.splits["cal"],
            alpha=alpha,
        )
        for score_name, key in [
            ("unweighted_l2_residual", "q_generic"),
            ("dual_weighted_lagrangian", "q_decision"),
            ("value_direction", "q_value"),
            ("delivery_direction", "q_delivery"),
            ("budget_direction", "q_budget"),
            ("member_direction", "q_member"),
        ]:
            rows.append(
                {
                    "case": case_name,
                    "model": model,
                    "score": score_name,
                    "radius": radii[key],
                    "lambda_delivery": radii["lambda_delivery"],
                    "lambda_budget": radii["lambda_budget"],
                    "lambda_member": radii["lambda_member"],
                    "lambda_delivery_sd": radii["lambda_delivery_sd"],
                    "lambda_budget_sd": radii["lambda_budget_sd"],
                    "lambda_member_sd": radii["lambda_member_sd"],
                    "component_alpha": radii.get("component_alpha", np.nan),
                    "dual_mode": radii.get("dual_mode", ""),
                }
            )
    return pd.DataFrame(rows)


def calibration_sample_size_diagnostic(
    case_name: str = "criteo",
    model: str = "numpy_mlp",
    max_rows: int = 500_000,
    seed: int = 13,
    n_blocks: int = 480,
    alpha: float = 0.1,
    alpha_tau: float = 0.1,
    alpha_m: float = 0.1,
    n_repeats: int = 48,
    cal_sizes: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Empirically test Proposition 4.2's calibration-size stability logic.

    The diagnostic uses the same public-data-calibrated streaming case, fitted
    forecaster, component support-function radii, causal/member uncertainty
    attachment, and robust selector used in the main experiments. It varies
    only the number of calibration blocks available to estimate the component
    quantiles.
    """
    bundle = build_case_bundle(
        case_name,
        max_rows=max_rows,
        seed=seed,
        n_blocks=n_blocks,
        model_names=[model],
    )
    if model not in bundle.forecasts:
        raise ValueError(f"Model {model!r} was not fitted.")
    response = causal_response_summary(case_name, max_rows=max_rows)
    _, z_hat = _planning_vectors(bundle, model)
    full_cal_idx = np.asarray(bundle.splits["cal"], dtype=int)
    z_test_blocks = bundle.case.z_blocks[bundle.splits["test"]]
    rng = np.random.default_rng(seed + 1701)

    reference_radii = _attach_effect_radii(
        calibrate_radii(
            bundle.policies,
            bundle.case,
            bundle.case.z_blocks,
            bundle.forecasts[model].predictions,
            full_cal_idx,
            alpha=alpha,
        ),
        bundle,
        response,
    )
    reference_scores, _ = robust_select(
        bundle.policies,
        bundle.case,
        z_hat,
        reference_radii,
        alpha_tau=alpha_tau,
        alpha_m=alpha_m,
    )
    reference_selected = reference_scores[reference_scores["selected"]]
    reference_has_policy = not reference_selected.empty
    if reference_has_policy:
        reference_policy = str(reference_selected.iloc[0]["policy"])
        reference_realized = _realized_policy_metrics(
            bundle.policies,
            bundle.case,
            z_test_blocks,
            reference_policy,
        )
    else:
        reference_policy = "unresolved"
        reference_realized = _empty_realized_metrics()

    feasible_ref = reference_scores[reference_scores["all_constraints_certified"]].copy()
    if reference_has_policy:
        selected_row = reference_selected.iloc[0]
        selected_slack = min(
            -float(selected_row["robust_delivery_shortfall"]),
            -float(selected_row["robust_budget_overspend"]),
            -float(selected_row["robust_member_violation"]),
        )
        if len(feasible_ref) > 1:
            feasible_sorted = feasible_ref.sort_values("robust_value", ascending=False)
            value_gap = float(feasible_sorted.iloc[0]["robust_value"] - feasible_sorted.iloc[1]["robust_value"])
        else:
            value_gap = np.inf
    else:
        selected_slack = np.nan
        value_gap = np.nan

    infeasible_ref = reference_scores[~reference_scores["all_constraints_certified"]].copy()
    if not infeasible_ref.empty:
        max_violation = infeasible_ref[
            [
                "robust_delivery_shortfall",
                "robust_budget_overspend",
                "robust_member_violation",
            ]
        ].max(axis=1)
        infeasible_separation = float(max_violation[max_violation > 0].min()) if np.any(max_violation > 0) else np.inf
    else:
        infeasible_separation = np.inf

    finite_margins = [
        x for x in [selected_slack, value_gap, infeasible_separation] if np.isfinite(x) and x >= 0
    ]
    decision_margin = float(min(finite_margins)) if finite_margins else np.nan

    reference_rows = [
        {
            "case": case_name,
            "model": model,
            "n_blocks": n_blocks,
            "n_full_calibration_blocks": len(full_cal_idx),
            "reference_policy": reference_policy,
            "reference_has_certified_policy": reference_has_policy,
            "reference_shortlist_size": int(reference_scores["shortlist"].sum()),
            "reference_q_decision": reference_radii["q_decision"],
            "reference_q_value": reference_radii["q_value"],
            "reference_q_delivery": reference_radii["q_delivery"],
            "reference_q_budget": reference_radii["q_budget"],
            "reference_q_member": reference_radii["q_member"],
            "reference_q_generic": reference_radii["q_generic"],
            "selected_feasibility_slack": selected_slack,
            "value_gap_to_runner_up": value_gap,
            "infeasible_policy_separation": infeasible_separation,
            "decision_margin": decision_margin,
            "half_margin": decision_margin / 2 if np.isfinite(decision_margin) else np.nan,
            "alpha": alpha,
            "alpha_tau": alpha_tau,
            "alpha_m": alpha_m,
            "component_alpha": reference_radii.get("component_alpha", np.nan),
            "tau_se": reference_radii.get("tau_se", np.nan),
            "member_se": reference_radii.get("member_se", np.nan),
            "true_value": reference_realized["true_value"],
            "any_violation_rate": reference_realized["any_violation_rate"],
        }
    ]

    if cal_sizes is None:
        base_sizes = [8, 12, 16, 24, 32, 48, 64, 96, len(full_cal_idx)]
        cal_sizes = sorted({s for s in base_sizes if 4 <= s <= len(full_cal_idx)})
    else:
        cal_sizes = sorted({int(s) for s in cal_sizes if 4 <= int(s) <= len(full_cal_idx)})

    component_keys = ["q_value", "q_delivery", "q_budget", "q_member"]
    rows = []
    for n_cal in cal_sizes:
        repeats = 1 if n_cal == len(full_cal_idx) else n_repeats
        for repeat in range(repeats):
            if n_cal == len(full_cal_idx):
                sample_idx = full_cal_idx
            else:
                sample_idx = np.sort(rng.choice(full_cal_idx, size=n_cal, replace=False))
            radii = _attach_effect_radii(
                calibrate_radii(
                    bundle.policies,
                    bundle.case,
                    bundle.case.z_blocks,
                    bundle.forecasts[model].predictions,
                    sample_idx,
                    alpha=alpha,
                ),
                bundle,
                response,
            )
            scores, _ = robust_select(
                bundle.policies,
                bundle.case,
                z_hat,
                radii,
                alpha_tau=alpha_tau,
                alpha_m=alpha_m,
            )
            selected = scores[scores["selected"]]
            has_policy = not selected.empty
            selected_policy = str(selected.iloc[0]["policy"]) if has_policy else "unresolved"
            component_errors = {
                f"abs_error_{key}": abs(float(radii[key]) - float(reference_radii[key]))
                for key in component_keys
            }
            max_component_error = float(max(component_errors.values()))
            theorem_margin_condition = bool(
                reference_has_policy
                and np.isfinite(decision_margin)
                and max_component_error <= decision_margin / 2
            )
            same_reference_policy = (
                bool(has_policy and selected_policy == reference_policy)
                if reference_has_policy
                else np.nan
            )
            same_decision_state = bool(
                (reference_has_policy and same_reference_policy)
                or ((not reference_has_policy) and (not has_policy))
            )
            if has_policy:
                realized = _realized_policy_metrics(bundle.policies, bundle.case, z_test_blocks, selected_policy)
            else:
                realized = _empty_realized_metrics()
            rows.append(
                {
                    "case": case_name,
                    "model": model,
                    "n_calibration_blocks": n_cal,
                    "repeat": repeat,
                    "selected_policy": selected_policy,
                    "has_certified_policy": has_policy,
                    "same_reference_policy": same_reference_policy,
                    "same_decision_state": same_decision_state,
                    "shortlist_size": int(scores["shortlist"].sum()),
                    "q_decision": radii["q_decision"],
                    "q_value": radii["q_value"],
                    "q_delivery": radii["q_delivery"],
                    "q_budget": radii["q_budget"],
                    "q_member": radii["q_member"],
                    "q_generic": radii["q_generic"],
                    "max_component_quantile_error": max_component_error,
                    "theorem_margin_condition": theorem_margin_condition,
                    "reference_policy": reference_policy,
                    "reference_has_certified_policy": reference_has_policy,
                    "decision_margin": decision_margin,
                    "half_margin": decision_margin / 2 if np.isfinite(decision_margin) else np.nan,
                    "component_alpha": radii.get("component_alpha", np.nan),
                    "alpha": alpha,
                    "alpha_tau": alpha_tau,
                    "alpha_m": alpha_m,
                    "tau_se": radii.get("tau_se", np.nan),
                    "member_se": radii.get("member_se", np.nan),
                    **component_errors,
                }
            )
            rows[-1].update(realized)
    return pd.DataFrame(rows), pd.DataFrame(reference_rows)
