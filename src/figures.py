from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd

from .config import FIGURE_DIR, ensure_artifact_dirs


def save_current(name: str) -> Path:
    ensure_artifact_dirs()
    path = FIGURE_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    return path


def plot_pacing_comparison(df: pd.DataFrame, name: str = "main_pacing_comparison.png") -> Path:
    """Plot main pacing results while explicitly showing unresolved decisions."""
    methods = [
        m
        for m in ["point_forecast", "generic_residual_conformal", "decision_calibrated"]
        if m in set(df["method"])
    ]
    methods += [m for m in df["method"].drop_duplicates().tolist() if m not in methods]
    cases = df["case"].drop_duplicates().tolist()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = {case: colors[i % len(colors)] for i, case in enumerate(cases)}

    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.4))

    def draw_panel(ax, value_col: str, title: str, ylabel: str) -> None:
        x = np.arange(len(methods))
        width = min(0.8 / max(len(cases), 1), 0.32)
        certified_values = pd.to_numeric(
            df.loc[df["has_certified_policy"].astype(bool), value_col],
            errors="coerce",
        ).dropna()
        scale = float(certified_values.max()) if not certified_values.empty else 1.0
        unresolved_height = max(0.06 * scale, 0.02)
        for j, case in enumerate(cases):
            offset = (j - (len(cases) - 1) / 2) * width
            heights = []
            certified = []
            labels = []
            for method in methods:
                row = df[(df["method"] == method) & (df["case"] == case)]
                if row.empty:
                    heights.append(0.0)
                    certified.append(False)
                    labels.append("missing")
                    continue
                r = row.iloc[0]
                is_certified = bool(r.get("has_certified_policy", False))
                value = pd.to_numeric(pd.Series([r.get(value_col)]), errors="coerce").iloc[0]
                heights.append(float(value) if is_certified and np.isfinite(value) else unresolved_height)
                certified.append(is_certified)
                labels.append("unresolved" if not is_certified else "")
            bars = ax.bar(
                x + offset,
                heights,
                width,
                label=case,
                color=color_map[case],
                edgecolor="black",
                linewidth=0.6,
            )
            for bar, is_certified, label in zip(bars, certified, labels):
                if not is_certified:
                    bar.set_hatch("//")
                    bar.set_alpha(0.28)
                    bar.set_facecolor("lightgray")
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        unresolved_height / 2,
                        label,
                        ha="center",
                        va="center",
                        rotation=0,
                        fontsize=8.4,
                        color="#4b5563",
                    )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("_", "\n") for m in methods], rotation=0)
        ax.margins(y=0.18)

    draw_panel(axes[0], "true_value", "Realized yield among certified policies", "true value")
    delivery_metric = "delivery_violation_rate" if "delivery_violation_rate" in df.columns else "delivery_violation"
    any_metric = "any_violation_rate" if "any_violation_rate" in df.columns else "any_violation"
    draw_panel(axes[1], delivery_metric, "Held-out underdelivery risk", "held-out block share")
    draw_panel(axes[2], any_metric, "Held-out any-constraint risk", "held-out block share")
    legend_handles = [Patch(facecolor=color_map[c], edgecolor="black", label=c) for c in cases]
    legend_handles.append(Patch(facecolor="lightgray", edgecolor="black", hatch="//", label="unresolved"))
    axes[2].legend(handles=legend_handles, loc="upper left", frameon=True)
    return save_current(name)


def plot_forecaster_comparison(df: pd.DataFrame, name: str = "main_forecaster_radius_yield.png") -> Path:
    """Show forecaster choice as a decision-radius and certification diagnostic."""
    model_names = {
        "point_naive": "naive",
        "seasonal_ridge": "seasonal\nridge",
        "random_feature": "random\nfeatures",
        "gradient_boosted": "gradient\nboosted",
        "numpy_mlp": "MLP",
        "torch_gru": "GRU",
        "torch_transformer": "Transformer",
    }
    df = df.copy()
    df["model_label"] = df["model"].map(model_names).fillna(df["model"].str.replace("_", "\n"))
    df["certified"] = df["has_certified_policy"].astype(bool)
    df["generic_ratio"] = df["q_generic"] / df["q_decision"].replace(0, np.nan)

    cases = df["case"].drop_duplicates().tolist()
    fig = plt.figure(figsize=(14.8, 8.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1.08], height_ratios=[1.0, 1.12])
    radius_axes = [fig.add_subplot(gs[i, 0]) for i in range(min(2, len(cases)))]
    ratio_ax = fig.add_subplot(gs[0, 1])
    status_ax = fig.add_subplot(gs[1, 1])

    certified_color = "#2563eb"
    unresolved_color = "#9ca3af"
    for ax, case in zip(radius_axes, cases):
        part = df[df["case"] == case].sort_values("q_decision", ascending=True).reset_index(drop=True)
        y = np.arange(len(part))
        colors = np.where(part["certified"], certified_color, unresolved_color)
        bars = ax.barh(y, part["q_decision"], color=colors, edgecolor="black", linewidth=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels(part["model_label"], fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel("decision-calibrated radius", fontsize=11)
        ax.set_title(f"{case}: smaller priced uncertainty is better", fontsize=13)
        xmax = max(float(part["q_decision"].max()) * 1.32, 1.0)
        ax.set_xlim(0, xmax)
        for bar, (_, row) in zip(bars, part.iterrows()):
            status = "certified" if row["certified"] else "unresolved"
            detail = (
                f"{row['q_decision']:.0f}, {status}"
                if row["q_decision"] >= 100
                else f"{row['q_decision']:.1f}, {status}"
            )
            ax.text(
                bar.get_width() + xmax * 0.015,
                bar.get_y() + bar.get_height() / 2,
                detail,
                va="center",
                fontsize=9.2,
                color="dimgray",
            )

    if len(cases) < 2:
        radius_axes[0].set_ylabel("forecaster")

    ratio_data = df.sort_values("generic_ratio", ascending=True)
    ratio_y = np.arange(len(ratio_data))
    ratio_colors = [certified_color if c else unresolved_color for c in ratio_data["certified"]]
    ratio_ax.barh(ratio_y, ratio_data["generic_ratio"], color=ratio_colors, edgecolor="black", linewidth=0.4)
    ratio_ax.set_yticks(ratio_y)
    ratio_ax.set_yticklabels(
        [f"{r.case}: {r.model_label.replace(chr(10), ' ')}" for r in ratio_data.itertuples()],
        fontsize=8.5,
    )
    ratio_ax.invert_yaxis()
    ratio_ax.set_xlabel("generic radius / decision radius", fontsize=11)
    ratio_ax.set_title("Generic calibration reserves much more slack", fontsize=13)
    ratio_ax.axvline(1.0, color="black", linestyle="--", linewidth=0.8)

    status_models = df.sort_values(["case", "q_decision"]).drop_duplicates("model")["model"].tolist()
    status_cases = cases
    status_ax.set_xlim(0, len(status_cases))
    status_ax.set_ylim(0, len(status_models))
    status_ax.set_xticks(np.arange(len(status_cases)) + 0.5)
    status_ax.set_xticklabels(status_cases, fontsize=10)
    status_ax.set_yticks(np.arange(len(status_models)) + 0.5)
    status_ax.set_yticklabels([model_names.get(m, m.replace("_", " ")) for m in status_models], fontsize=10)
    status_ax.invert_yaxis()
    status_ax.set_title("Certification outcome", fontsize=13)
    certified_summaries = []
    for i, model in enumerate(status_models):
        for j, case in enumerate(status_cases):
            row = df[(df["model"] == model) & (df["case"] == case)]
            if row.empty:
                continue
            r = row.iloc[0]
            certified = bool(r["certified"])
            color = certified_color if certified else unresolved_color
            rect = Rectangle(
                (j + 0.05, i + 0.08),
                0.90,
                0.84,
                facecolor=color,
                edgecolor="black",
                linewidth=0.7,
                alpha=0.95 if certified else 0.65,
            )
            status_ax.add_patch(rect)
            if certified:
                text = "certified"
                certified_summaries.append(
                    (
                        str(case),
                        str(r["selected_policy"]).replace("pace_", ""),
                        float(r["any_violation_rate"]),
                    )
                )
            else:
                text = "unresolved"
            status_ax.text(
                j + 0.5,
                i + 0.5,
                text,
                ha="center",
                va="center",
                fontsize=8.8,
                color="white",
                linespacing=0.95,
            )
    for j in range(len(status_cases) + 1):
        status_ax.axvline(j, color="white", linewidth=1.2)
    for i in range(len(status_models) + 1):
        status_ax.axhline(i, color="white", linewidth=1.2)
    status_ax.tick_params(length=0)
    if certified_summaries:
        summary = pd.DataFrame(
            certified_summaries,
            columns=["case", "policy", "violation"],
        ).drop_duplicates()
        summary_text = "; ".join(
            f"{row.case}: {row.policy}, violation={row.violation:.2f}"
            for row in summary.itertuples()
        )
        status_ax.text(
            0.5,
            -0.12,
            f"Certified cells share: {summary_text}",
            transform=status_ax.transAxes,
            ha="center",
            va="top",
            fontsize=9.2,
            color="dimgray",
        )
    fig.legend(
        handles=[
            Patch(facecolor=certified_color, edgecolor="black", label="certified policy"),
            Patch(facecolor=unresolved_color, edgecolor="black", label="unresolved"),
        ],
        loc="lower center",
        ncol=2,
        frameon=True,
    )
    return save_current(name)


def plot_causal_modes(df: pd.DataFrame, name: str = "main_causal_response_modes.png") -> Path:
    mode_names = {
        "predicted_response": "predicted",
        "doubly_robust_response": "doubly\nrobust",
        "robust_causal_response": "robust\ncausal",
    }
    modes = [m for m in mode_names if m in set(df["mode"])]
    modes += [m for m in df["mode"].drop_duplicates().tolist() if m not in modes]
    cases = df["case"].drop_duplicates().tolist()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = {case: colors[i % len(colors)] for i, case in enumerate(cases)}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(len(modes))
    width = min(0.8 / max(len(cases), 1), 0.32)
    for j, case in enumerate(cases):
        part = df[df["case"] == case].set_index("mode")
        offset = (j - (len(cases) - 1) / 2) * width
        response_vals = [float(part.loc[m, "response_estimate"]) if m in part.index else np.nan for m in modes]
        axes[0].bar(
            x + offset,
            response_vals,
            width,
            color=color_map[case],
            edgecolor="black",
            linewidth=0.6,
            label=case,
        )
        heights = []
        certified = []
        unresolved_height = 24.0
        for m in modes:
            if m not in part.index:
                heights.append(0.0)
                certified.append(False)
                continue
            row = part.loc[m]
            ok = bool(row["has_certified_policy"])
            val = pd.to_numeric(pd.Series([row.get("true_value")]), errors="coerce").iloc[0]
            heights.append(float(val) if ok and np.isfinite(val) else unresolved_height)
            certified.append(ok)
        bars = axes[1].bar(
            x + offset,
            heights,
            width,
            color=color_map[case],
            edgecolor="black",
            linewidth=0.6,
            label=case,
        )
        for bar, ok, m in zip(bars, certified, modes):
            row = part.loc[m] if m in part.index else None
            if not ok:
                bar.set_alpha(0.30)
                bar.set_hatch("//")
                bar.set_facecolor("lightgray")
                axes[1].text(
                    bar.get_x() + bar.get_width() / 2,
                    unresolved_height / 2,
                    "unresolved",
                    ha="center",
                    va="center",
                    rotation=0,
                    fontsize=8.0,
                    color="dimgray",
                )
            elif row is not None:
                axes[1].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"viol={float(row['any_violation_rate']):.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    color="dimgray",
                )

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([mode_names.get(m, m.replace("_", "\n")) for m in modes])
        ax.margins(y=0.18)
    axes[0].set_ylabel("response estimate")
    axes[0].set_title("Causal response input to pacing")
    axes[1].set_ylabel("realized yield among certified policies")
    axes[1].set_title("Certified yield and unresolved modes")
    axes[1].set_ylim(0, max(axes[1].get_ylim()[1], 430))
    legend_handles = [Patch(facecolor=color_map[c], edgecolor="black", label=c) for c in cases]
    legend_handles.append(Patch(facecolor="lightgray", edgecolor="black", hatch="//", label="unresolved"))
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=3,
        frameon=True,
    )
    return save_current(name)


def plot_calibration_sample_size(
    df: pd.DataFrame,
    reference: pd.DataFrame,
    name: str = "appendix_calibration_sample_size.png",
) -> Path:
    """Plot the calibration-block sample-size diagnostic for Proposition 4.2."""
    cases = df["case"].drop_duplicates().tolist()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = {case: colors[i % len(colors)] for i, case in enumerate(cases)}

    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.6))
    for case in cases:
        part = df[df["case"] == case].copy()
        grouped = (
            part.groupby("n_calibration_blocks")
            .agg(
                median_error=("max_component_quantile_error", "median"),
                q25_error=("max_component_quantile_error", lambda x: np.quantile(x, 0.25)),
                q75_error=("max_component_quantile_error", lambda x: np.quantile(x, 0.75)),
                same_state_rate=("same_decision_state", "mean"),
                same_policy_rate=("same_reference_policy", "mean"),
                margin_condition_rate=("theorem_margin_condition", "mean"),
                median_shortlist=("shortlist_size", "median"),
            )
            .reset_index()
        )
        x = grouped["n_calibration_blocks"].to_numpy(dtype=float)
        color = color_map[case]
        positive_errors = grouped[
            ["median_error", "q25_error", "q75_error"]
        ].to_numpy(dtype=float)
        positive_errors = positive_errors[positive_errors > 0]
        error_floor = float(max(np.min(positive_errors) * 0.5, 1e-3)) if positive_errors.size else 1e-3
        median_error = np.maximum(grouped["median_error"].to_numpy(dtype=float), error_floor)
        q25_error = np.maximum(grouped["q25_error"].to_numpy(dtype=float), error_floor)
        q75_error = np.maximum(grouped["q75_error"].to_numpy(dtype=float), error_floor)
        axes[0].plot(x, median_error, marker="o", color=color, label=case)
        axes[0].fill_between(
            x,
            q25_error,
            q75_error,
            color=color,
            alpha=0.15,
            linewidth=0,
        )
        ref_row = reference[reference["case"] == case]
        if not ref_row.empty:
            half_margin = float(ref_row.iloc[0]["half_margin"])
            if np.isfinite(half_margin):
                axes[0].axhline(half_margin, color=color, linestyle="--", linewidth=1.0, alpha=0.75)
        axes[1].plot(
            x,
            grouped["same_state_rate"],
            marker="o",
            color=color,
            label=f"{case}: same state",
        )
        axes[1].plot(
            x,
            grouped["margin_condition_rate"],
            marker="s",
            color=color,
            linestyle="--",
            alpha=0.85,
            label=f"{case}: margin condition",
        )
        axes[2].plot(x, grouped["median_shortlist"], marker="o", color=color, label=case)

    axes[0].set_title("Calibration error generally falls with more data")
    axes[0].set_xlabel("calibration blocks")
    axes[0].set_ylabel("max component quantile error")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.25)
    axes[0].text(
        0.02,
        0.96,
        "zero reference errors shown at floor",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        color="dimgray",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
    )

    axes[1].set_title("Stable decision when error clears margin")
    axes[1].set_xlabel("calibration blocks")
    axes[1].set_ylabel("share over resamples")
    axes[1].set_xscale("log")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].grid(alpha=0.25)

    axes[2].set_title("Shortlist size under calibration uncertainty")
    axes[2].set_xlabel("calibration blocks")
    axes[2].set_ylabel("median shortlist size")
    axes[2].set_xscale("log")
    axes[2].grid(alpha=0.25)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=4, frameon=True)
    return save_current(name)


def plot_theory_geometry(separation: pd.DataFrame, coverage: pd.DataFrame, name: str = "appendix_theory_geometry.png") -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(separation["nuisance_dimension"], separation["q_decision"], marker="o", label="decision")
    axes[0].plot(separation["nuisance_dimension"], separation["q_generic"], marker="o", label="generic")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("nuisance dimension")
    axes[0].set_ylabel("radius")
    axes[0].set_title("High-dimensional separation")
    axes[0].legend()
    axes[1].plot(coverage["dimension"], coverage["decision_coverage"], marker="o", label="decision")
    axes[1].plot(coverage["dimension"], coverage["generic_coverage"], marker="o", label="generic")
    axes[1].axhline(coverage["target_coverage"].iloc[0], color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("dimension")
    axes[1].set_ylabel("coverage")
    axes[1].set_title("Split conformal coverage")
    axes[1].legend()
    return save_current(name)


def plot_support_characterization(
    characterization: pd.DataFrame,
    name: str = "appendix_support_characterization.png",
) -> Path:
    """Visualize the support-function characterization diagnostic."""
    df = characterization.copy()
    labels = {
        "minimal_signed_hull": "minimal\nsigned hull",
        "l2_ball_superset": "L2 ball\nsuperset",
        "coordinate_box_superset": "coordinate box\nsuperset",
        "missing_active_sensitivity": "missing active\nsensitivity",
        "understated_support": "understated\nsupport",
    }
    colors = np.where(df["valid_for_catalog"].astype(bool), "#2563eb", "#dc2626")
    x = np.arange(len(df))

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2), constrained_layout=True)
    axes[0].bar(
        x,
        df["mean_certificate_ratio"],
        color=colors,
        edgecolor="black",
        linewidth=0.7,
    )
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="minimal support")
    axes[0].set_ylabel("mean certificate / minimal support")
    axes[0].set_title("Certificate size")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([labels.get(c, c.replace("_", "\n")) for c in df["certificate"]], fontsize=9)
    axes[0].set_ylim(0, max(1.15, float(df["mean_certificate_ratio"].max()) * 1.18))
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(
        x,
        df["violation_rate"],
        color=colors,
        edgecolor="black",
        linewidth=0.7,
    )
    axes[1].set_ylabel("catalog violation rate")
    axes[1].set_title("Validity check")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([labels.get(c, c.replace("_", "\n")) for c in df["certificate"]], fontsize=9)
    axes[1].set_ylim(0, 1.08)
    axes[1].grid(axis="y", alpha=0.25)

    legend_handles = [
        Patch(facecolor="#2563eb", edgecolor="black", label="valid certificate"),
        Patch(facecolor="#dc2626", edgecolor="black", label="invalid certificate"),
    ]
    axes[1].legend(handles=legend_handles, loc="upper left", frameon=True)
    return save_current(name)


def plot_ablation(df: pd.DataFrame, x: str, y: str, hue: str, title: str, name: str) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, part in df.groupby(hue):
        ax.plot(part[x], part[y], marker="o", label=str(key))
    ax.set_xlabel(x.replace("_", " "))
    ax.set_ylabel(y.replace("_", " "))
    ax.set_title(title)
    ax.legend()
    return save_current(name)
