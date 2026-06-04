from __future__ import annotations

import numpy as np
import pandas as pd

from .forecasting import conformal_quantile
from .pacing import decision_scores, generic_scores


def sharp_support_function_check(seed: int = 1, n_policies: int = 8, dim: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(n_policies, dim))
    errors = rng.normal(size=(200, dim))
    phi = decision_scores(w, errors)
    direct = np.max(np.abs(errors @ w.T), axis=1)
    understated = 0.8 * phi
    violation = direct > understated + 1e-12
    return pd.DataFrame(
        {
            "check": ["support_equality", "minimality_violation"],
            "max_abs_gap": [float(np.max(np.abs(phi - direct))), float(np.max(direct - understated))],
            "violation_rate": [0.0, float(violation.mean())],
        }
    )


def support_function_characterization_check(
    seed: int = 11,
    n_policies: int = 8,
    dim: int = 16,
    n_errors: int = 500,
) -> pd.DataFrame:
    """Check the signed-hull characterization behind Theorem 4.1.

    The minimal certificate is the support function of the signed policy
    sensitivity hull. Supersets of that hull are valid but larger. Sets that
    exclude an active sensitivity, or scalar multiples below the support
    function, fail on constructed and random forecast-error directions.
    """
    if dim < n_policies:
        raise ValueError("dim must be at least n_policies for the orthogonal active-sensitivity check.")
    rng = np.random.default_rng(seed)
    w = np.eye(dim)[:n_policies]
    random_errors = rng.normal(size=(n_errors, dim))
    targeted_errors = np.vstack([np.eye(dim)[0], -np.eye(dim)[0], 2.0 * np.eye(dim)[0]])
    errors = np.vstack([random_errors, targeted_errors])

    minimal = decision_scores(w, errors)
    l2_radius = float(np.max(np.linalg.norm(w, axis=1)))
    l2_superset = l2_radius * np.linalg.norm(errors, axis=1)
    coordinate_box = np.max(np.abs(w), axis=0)
    box_superset = np.abs(errors) @ coordinate_box
    missing_active = decision_scores(w[1:], errors)
    understated = 0.8 * minimal

    certificates = [
        (
            "minimal_signed_hull",
            "equal_to_signed_hull",
            True,
            True,
            minimal,
        ),
        (
            "l2_ball_superset",
            "strict_superset",
            True,
            True,
            l2_superset,
        ),
        (
            "coordinate_box_superset",
            "strict_superset",
            True,
            True,
            box_superset,
        ),
        (
            "missing_active_sensitivity",
            "does_not_contain_signed_hull",
            False,
            True,
            missing_active,
        ),
        (
            "understated_support",
            "scaled_below_signed_hull",
            False,
            False,
            understated,
        ),
    ]
    rows = []
    positive = minimal > 1e-12
    for name, relation, contains_signed_hull, is_support_function, cert in certificates:
        shortfall = np.maximum(minimal - cert, 0.0)
        excess = np.maximum(cert - minimal, 0.0)
        ratio = np.full_like(minimal, np.nan, dtype=float)
        ratio[positive] = cert[positive] / minimal[positive]
        rows.append(
            {
                "certificate": name,
                "set_relation_to_signed_hull": relation,
                "contains_signed_sensitivities": contains_signed_hull,
                "is_support_function_certificate": is_support_function,
                "valid_for_catalog": bool(np.all(shortfall <= 1e-12)),
                "max_excess_over_minimal": float(np.max(excess)),
                "max_shortfall_below_minimal": float(np.max(shortfall)),
                "violation_rate": float(np.mean(shortfall > 1e-12)),
                "mean_certificate_ratio": float(np.nanmean(ratio)),
            }
        )
    return pd.DataFrame(rows)


def conformal_coverage_check(seed: int = 2, n_cal: int = 88, n_trials: int = 20000, alpha: float = 0.1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for dim in [6, 24, 96]:
        w = rng.normal(size=(10, dim))
        covered_dc = []
        covered_generic = []
        q_dc_values = []
        q_gen_values = []
        for _ in range(n_trials):
            errors = rng.normal(size=(n_cal + 1, dim))
            cal, test = errors[:-1], errors[-1:]
            q_dc = conformal_quantile(decision_scores(w, cal), alpha=alpha)
            q_gen = conformal_quantile(generic_scores(cal), alpha=alpha)
            covered_dc.append(decision_scores(w, test)[0] <= q_dc)
            covered_generic.append(generic_scores(test)[0] <= q_gen)
            q_dc_values.append(q_dc)
            q_gen_values.append(q_gen)
        rows.append(
            {
                "dimension": dim,
                "target_coverage": 1 - alpha,
                "decision_coverage": float(np.mean(covered_dc)),
                "generic_coverage": float(np.mean(covered_generic)),
                "mean_q_decision": float(np.mean(q_dc_values)),
                "mean_q_generic": float(np.mean(q_gen_values)),
            }
        )
    return pd.DataFrame(rows)


def high_dimensional_separation_grid(max_m: int = 256, sigma: float = 1.0) -> pd.DataFrame:
    rows = []
    for m in [1, 2, 4, 8, 16, 32, 64, 128, max_m]:
        q_dc = 1.0
        q_gen = float(np.sqrt(1 + m * sigma**2))
        rows.append(
            {
                "nuisance_dimension": m,
                "q_decision": q_dc,
                "q_generic": q_gen,
                "generic_to_decision_ratio": q_gen / q_dc,
            }
        )
    return pd.DataFrame(rows)


def slack_necessity_check() -> pd.DataFrame:
    rows = []
    delta = 0.2
    for margin in [0.05, 0.15, 0.2, 0.25, 0.4]:
        nominal_g = -margin
        worst_case_g = nominal_g + delta
        rows.append(
            {
                "nominal_margin": margin,
                "decision_relevant_uncertainty_delta": delta,
                "worst_case_constraint": worst_case_g,
                "violates_without_slack": worst_case_g > 0,
                "certifiable_with_delta_slack": margin >= delta,
            }
        )
    return pd.DataFrame(rows)


def finite_catalog_check(seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dense = np.linspace(0.05, 1.25, 1000)

    def risk(x: np.ndarray) -> np.ndarray:
        return (x - 0.72) ** 2 + 0.08 * np.abs(np.sin(8 * x)) + 0.04 * x

    r_dense = risk(dense)
    best_dense = float(r_dense.min())
    rows = []
    for n in [4, 6, 8, 12, 20, 40]:
        catalog = np.linspace(0.05, 1.25, n)
        eta = float(np.max(np.min(np.abs(dense[:, None] - catalog[None, :]), axis=1)))
        best_catalog = float(risk(catalog).min())
        lip = 2.5
        rows.append(
            {
                "catalog_size": n,
                "eta": eta,
                "best_dense_risk": best_dense,
                "best_catalog_risk": best_catalog,
                "approximation_error": best_catalog - best_dense,
                "lipschitz_net_bound": lip * eta,
                "bound_holds": best_catalog - best_dense <= lip * eta + 1e-12,
            }
        )
    return pd.DataFrame(rows)


def robust_certificate_simulation(
    seed: int = 4,
    n_trials: int = 500,
    n_cal: int = 80,
    alpha: float = 0.1,
    alpha_tau: float = 0.1,
    alpha_m: float = 0.1,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    alpha_component = alpha / 4
    for dim in [8, 32, 96]:
        violations = []
        regrets = []
        regret_bounds = []
        regret_bound_holds = []
        unresolved = []
        coverage_events = []
        for _ in range(n_trials):
            n_policies = 10
            n_constraints = 3
            value_w = rng.normal(scale=0.15, size=(n_policies, dim))
            constraint_w = rng.normal(scale=0.04, size=(n_constraints, n_policies, dim))
            cal_errors = rng.normal(size=(n_cal, dim))
            test_error = rng.normal(size=(1, dim))

            q_value = conformal_quantile(decision_scores(value_w, cal_errors), alpha=alpha_component)
            q_constraints = []
            constraint_covered = []
            for j in range(n_constraints):
                q_j = conformal_quantile(decision_scores(constraint_w[j], cal_errors), alpha=alpha_component)
                q_constraints.append(q_j)
                constraint_covered.append(decision_scores(constraint_w[j], test_error)[0] <= q_j)
            q_constraints = np.asarray(q_constraints)
            value_covered = decision_scores(value_w, test_error)[0] <= q_value

            response_sigma = rng.uniform(0.02, 0.07, size=n_policies)
            cal_response_errors = rng.normal(scale=response_sigma[None, :], size=(n_cal, n_policies))
            test_response_errors = rng.normal(scale=response_sigma, size=n_policies)
            q_response = conformal_quantile(np.max(np.abs(cal_response_errors), axis=1), alpha=alpha_tau)
            response_radius = np.repeat(q_response, n_policies)
            response_covered = bool(np.all(np.abs(test_response_errors) <= response_radius))

            member_sigma = rng.uniform(0.02, 0.07, size=(n_policies, n_constraints))
            cal_member_errors = rng.normal(
                scale=member_sigma[None, :, :],
                size=(n_cal, n_policies, n_constraints),
            )
            test_member_errors = rng.normal(scale=member_sigma, size=(n_policies, n_constraints))
            q_member = conformal_quantile(
                np.max(np.abs(cal_member_errors), axis=(1, 2)),
                alpha=alpha_m,
            )
            member_radius = np.repeat(q_member, n_policies * n_constraints).reshape(n_policies, n_constraints)
            member_covered = bool(np.all(np.abs(test_member_errors) <= member_radius))

            coverage_events.append(
                bool(value_covered and np.all(constraint_covered) and response_covered and member_covered)
            )

            nominal_value = rng.normal(loc=1.0, scale=0.25, size=n_policies)
            nominal_constraints = rng.normal(loc=-1.4, scale=0.45, size=(n_policies, n_constraints))
            robust_value = nominal_value - q_value - response_radius
            robust_constraints = nominal_constraints + q_constraints[None, :] + member_radius
            feasible = np.all(robust_constraints <= 0, axis=1)
            unresolved.append(not np.any(feasible))
            if not np.any(feasible):
                violations.append(False)
                regrets.append(np.nan)
                regret_bounds.append(np.nan)
                regret_bound_holds.append(np.nan)
                continue

            selected = int(np.argmax(np.where(feasible, robust_value, -np.inf)))
            true_constraints = nominal_constraints.copy()
            true_value = nominal_value + (test_error @ value_w.T).ravel() + test_response_errors
            for j in range(n_constraints):
                true_constraints[:, j] += (test_error @ constraint_w[j].T).ravel()
            true_constraints += test_member_errors
            violations.append(bool(np.any(true_constraints[selected] > 0)))

            robust_best = int(np.argmax(np.where(feasible, true_value, -np.inf)))
            regret = float(max(0.0, true_value[robust_best] - true_value[selected]))
            bound = float(2.0 * np.max(q_value + response_radius))
            regrets.append(regret)
            regret_bounds.append(bound)
            regret_bound_holds.append(regret <= bound + 1e-12)
        rows.append(
            {
                "dimension": dim,
                "alpha": alpha,
                "alpha_tau": alpha_tau,
                "alpha_m": alpha_m,
                "component_alpha": alpha_component,
                "joint_coverage_rate": float(np.mean(coverage_events)),
                "target_joint_coverage": 1 - alpha - alpha_tau - alpha_m,
                "certificate_violation_rate": float(np.mean(violations)),
                "certified_rate": float(1.0 - np.mean(unresolved)),
                "unresolved_rate": float(np.mean(unresolved)),
                "mean_regret_proxy": float(np.nanmean(regrets)) if np.any(np.isfinite(regrets)) else np.nan,
                "mean_regret_bound": float(np.nanmean(regret_bounds)) if np.any(np.isfinite(regret_bounds)) else np.nan,
                "regret_bound_holds_rate": float(np.nanmean(regret_bound_holds))
                if np.any(np.isfinite(regret_bound_holds))
                else np.nan,
                "target_alpha": alpha,
            }
        )
    return pd.DataFrame(rows)


def theory_to_artifact_map() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "paper_result": "Sharp and calibrated decision radius",
                "code_check": "support equality, signed-hull characterization, split-conformal coverage",
                "primary_notebook": "04_appendix_theory_geometry_calibration.ipynb",
            },
            {
                "paper_result": "Calibration sample size for stable pacing certification",
                "code_check": "component quantile error, robust-selector stability, and empirical margin condition under calibration-block resampling",
                "primary_notebook": "07_appendix_calibration_sample_size.ipynb",
            },
            {
                "paper_result": "High-dimensional separation from generic residual calibration",
                "code_check": "generic radius grows with nuisance dimensions while decision radius stays fixed",
                "primary_notebook": "04_appendix_theory_geometry_calibration.ipynb",
            },
            {
                "paper_result": "Robust causal pacing certificate",
                "code_check": "robust selector violation and regret diagnostics",
                "primary_notebook": "05_appendix_certificate_slack_catalog.ipynb",
            },
            {
                "paper_result": "Appendix supporting theorem: decision-calibrated slack necessity",
                "code_check": "point-forecast margin below directional uncertainty causes violation",
                "primary_notebook": "05_appendix_certificate_slack_catalog.ipynb",
            },
            {
                "paper_result": "Appendix supporting proposition: finite catalog sufficiency",
                "code_check": "catalog approximation error remains below Lipschitz-net bound",
                "primary_notebook": "05_appendix_certificate_slack_catalog.ipynb",
            },
        ]
    )
