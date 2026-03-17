"""
Benchmark Data Analysis & Publication Figures

Generates all figures and summary statistics from benchmark results:
  - Diagnostic figures (bar charts, scatter, heatmap, lock-match)
  - Publication figures (CDF, violin, lock-in dynamics)
  - Progress metric figures (confusion gain, confidence drop, peak adversarial)
    — generated automatically when robust benchmark columns are present

Usage:
    python analyze_benchmark.py                          # Standard models (default)
    python analyze_benchmark.py --source robust          # Robust models
    python analyze_benchmark.py --skip-replay            # Skip lock-in replay (no GPU)
    python analyze_benchmark.py --show                   # Interactive display
    python analyze_benchmark.py --csv results/custom.csv --source robust
"""

import argparse
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


# ===========================================================================
# Style configuration
# ===========================================================================
def _setup_style():
    """Configure matplotlib for publication-quality output."""
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("seaborn-whitegrid")

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "figure.constrained_layout.use": True,
    })

    # Try LaTeX rendering; fall back silently
    try:
        from matplotlib.texmanager import TexManager
        TexManager._run_checked_subprocess(
            ["latex", "--version"], "latex")
        plt.rcParams["text.usetex"] = True
    except Exception:
        plt.rcParams["text.usetex"] = False


# ===========================================================================
# Color palettes
# ===========================================================================

# 3-mode palette (diagnostic figures: untargeted / targeted-oracle / opportunistic)
MODE_COLORS = {
    "untargeted": "#4878CF",     # blue
    "targeted": "#E8873A",       # orange
    "opportunistic": "#6BA353",  # green
}
MODE_ORDER = ["untargeted", "targeted", "opportunistic"]
MODE_LABELS = {
    "untargeted": "Untargeted",
    "targeted": "Targeted (oracle)",
    "opportunistic": "Opportunistic",
}

# 2-mode palette (publication figures: untargeted vs opportunistic)
COLOR_UNTARGETED = "#4878CF"      # blue
COLOR_OPPORTUNISTIC = "#D64541"   # red
LINESTYLE_SIMBA = "-"             # solid
LINESTYLE_SQUARE = "--"           # dashed

METHOD_MARKERS = {"SimBA": "o", "SquareAttack": "s"}
MODEL_ORDERS = {
    "standard": ["resnet18", "resnet50", "vgg16", "alexnet"],
    "robust": ["Salman2020Do_R18", "Salman2020Do_R50"],
}
STABILITY_THRESHOLDS = {"standard": 5, "robust": 10}
CASE_MODELS = {"standard": "resnet50", "robust": "Salman2020Do_R18"}


# ===========================================================================
# Data loading
# ===========================================================================
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["iterations"] = pd.to_numeric(df["iterations"])
    df["success"] = df["success"].map(
        {"True": True, "False": False, True: True, False: False}
    )
    df["switch_iteration"] = pd.to_numeric(df["switch_iteration"], errors="coerce")
    df["locked_class"] = pd.to_numeric(df["locked_class"], errors="coerce")
    df["oracle_target"] = pd.to_numeric(df["oracle_target"], errors="coerce")
    df["adversarial_class"] = pd.to_numeric(df["adversarial_class"], errors="coerce")
    df["mode"] = pd.Categorical(df["mode"], categories=MODE_ORDER, ordered=True)

    return df


# ===========================================================================
# Helpers
# ===========================================================================
def _savefig(fig, outdir: str, name: str):
    fig.savefig(os.path.join(outdir, f"{name}.png"), bbox_inches="tight")
    fig.savefig(os.path.join(outdir, f"{name}.pdf"), bbox_inches="tight")
    print(f"  Saved {name}.png / .pdf")


def _ci95(series):
    """Return 95% CI half-width."""
    n = len(series)
    if n < 2:
        return 0.0
    return stats.t.ppf(0.975, n - 1) * series.std(ddof=1) / np.sqrt(n)


def _annotate_sig_brackets(ax, test_results: pd.DataFrame, model: str | None,
                           method_list: list[str], x_positions: np.ndarray,
                           bar_width: float, y_data: dict):
    """Draw significance brackets between bars on a grouped bar chart.

    Args:
        ax: Matplotlib axes.
        test_results: DataFrame from compute_paired_tests().
        model: Model name to filter tests (None = all models pooled).
        method_list: Ordered method names matching x_positions.
        x_positions: X-coordinates of method groups.
        bar_width: Width of individual bars.
        y_data: Dict of {(method, mode): bar_height} for bracket positioning.
    """
    if test_results is None or test_results.empty:
        return

    # Mode index within group: untargeted=0, targeted=1, opportunistic=2
    mode_idx = {"untargeted": 0, "targeted": 1, "opportunistic": 2}
    # Only draw key comparisons: unt↔opp and unt↔oracle
    show_pairs = [("untargeted", "opportunistic"), ("untargeted", "targeted")]

    for j, method in enumerate(method_list):
        if model is not None:
            sub = test_results[(test_results["model"] == model) &
                               (test_results["method"] == method)]
        else:
            sub = test_results[test_results["method"] == method]

        # Determine max bar height for this method group
        max_y = max((y_data.get((method, m), 0) for m in MODE_ORDER), default=0)
        bracket_offset = 0

        for modeA, modeB in show_pairs:
            row = sub[(sub["modeA"] == modeA) & (sub["modeB"] == modeB)]
            if row.empty:
                continue
            row = row.iloc[0]

            sig = row["sig"]
            if sig == "ns":
                continue

            x1 = x_positions[j] + (mode_idx[modeA] - 1) * bar_width
            x2 = x_positions[j] + (mode_idx[modeB] - 1) * bar_width

            # Stack brackets vertically
            y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
            h = y_range * 0.03
            y = max_y + y_range * 0.06 + bracket_offset
            bracket_offset += y_range * 0.08

            ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y],
                    lw=0.8, color="0.3")
            ax.text((x1 + x2) / 2, y + h, sig,
                    ha="center", va="bottom", fontsize=8, color="0.3")

    # Expand y-limit to fit brackets
    if bracket_offset > 0:
        cur_top = ax.get_ylim()[1]
        needed = max_y + bracket_offset + ax.get_ylim()[1] * 0.08
        if needed > cur_top:
            ax.set_ylim(top=needed)


def _pub_label(method: str, mode: str) -> str:
    """Legend label for publication 2-mode figures."""
    short = "SimBA" if method == "SimBA" else "Square"
    tag = "Untargeted" if mode == "untargeted" else "Opportunistic"
    return f"{short} — {tag}"


def _pub_color(mode: str) -> str:
    return COLOR_UNTARGETED if mode == "untargeted" else COLOR_OPPORTUNISTIC


def _pub_linestyle(method: str) -> str:
    return LINESTYLE_SIMBA if method == "SimBA" else LINESTYLE_SQUARE


# ===========================================================================
# Diagnostic Figures (3-mode: untargeted / targeted-oracle / opportunistic)
# ===========================================================================

def fig_headline_bars(df: pd.DataFrame, outdir: str, test_results=None):
    """Headline bar chart: mean iterations by mode."""
    ok = df
    if ok.empty:
        print("  Skipping fig_headline_bars: no successful runs")
        return None
    agg = ok.groupby(["method", "mode"], observed=True)["iterations"].agg(["mean", "count", "std"])
    agg["ci"] = ok.groupby(["method", "mode"], observed=True)["iterations"].apply(_ci95)
    agg = agg.reset_index()

    methods = ["SimBA", "SquareAttack"]
    x = np.arange(len(methods))
    width = 0.22

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, mode in enumerate(MODE_ORDER):
        subset = agg[agg["mode"] == mode].set_index("method").reindex(methods)
        ax.bar(
            x + (i - 1) * width,
            subset["mean"],
            width,
            yerr=subset["ci"],
            capsize=4,
            color=MODE_COLORS[mode],
            label=MODE_LABELS[mode],
            edgecolor="white",
            linewidth=0.5,
        )

    # Annotate % savings (opportunistic vs untargeted)
    for j, method in enumerate(methods):
        u = agg[(agg["method"] == method) & (agg["mode"] == "untargeted")]["mean"].values
        o = agg[(agg["method"] == method) & (agg["mode"] == "opportunistic")]["mean"].values
        if len(u) and len(o) and u[0] > 0:
            savings = (u[0] - o[0]) / u[0] * 100
            y_pos = max(u[0], o[0]) + agg[(agg["method"] == method)]["ci"].max() + 80
            ax.annotate(
                f"\u2193 {savings:.1f}%",
                xy=(x[j] + width, y_pos),
                ha="center",
                fontsize=11,
                fontweight="bold",
                color=MODE_COLORS["opportunistic"],
            )

    # Significance brackets (pooled: pair on model+image, one test per method)
    if test_results is not None and not test_results.empty:
        y_data = {}
        for j, method in enumerate(methods):
            for mode in MODE_ORDER:
                row = agg[(agg["method"] == method) & (agg["mode"] == mode)]
                if not row.empty:
                    y_data[(method, mode)] = row["mean"].values[0] + row["ci"].values[0]
        pooled_tests = compute_paired_tests(
            ok, metric="iterations", filter_success=True, pool_models=True)
        _annotate_sig_brackets(ax, pooled_tests, "_pooled_",
                               methods, x, width, y_data)

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Mean Iterations")
    ax.set_title("Mean Iterations by Attack Mode (All Models, All Runs)")
    ax.legend()
    ax.set_ylim(bottom=0)
    _savefig(fig, outdir, "fig_headline_bars")
    return fig


def fig_per_model(df: pd.DataFrame, outdir: str, model_order: list[str],
                  test_results=None):
    """Per-model breakdown: mean iterations by mode."""
    ok = df
    if ok.empty:
        print("  Skipping fig_per_model: no successful runs")
        return None
    models = [m for m in model_order if m in ok["model"].unique()]
    methods = ["SimBA", "SquareAttack"]

    with plt.rc_context({"figure.constrained_layout.use": False}):
        fig, axes = plt.subplots(
            1, len(models), figsize=(4 * len(models), 5), sharey=False,
        )
    fig.subplots_adjust(bottom=0.15, top=0.88, wspace=0.3)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        sub = ok[ok["model"] == model]
        agg = sub.groupby(["method", "mode"], observed=True)["iterations"].agg(["mean"]).reset_index()
        agg["ci"] = sub.groupby(["method", "mode"], observed=True)["iterations"].apply(_ci95).values

        x = np.arange(len(methods))
        width = 0.22
        for i, mode in enumerate(MODE_ORDER):
            m_data = agg[agg["mode"] == mode].set_index("method").reindex(methods)
            ax.bar(
                x + (i - 1) * width,
                m_data["mean"],
                width,
                yerr=m_data["ci"],
                capsize=3,
                color=MODE_COLORS[mode],
                edgecolor="white",
                linewidth=0.5,
            )
        # Significance brackets
        if test_results is not None and not test_results.empty:
            y_data = {}
            for j2, meth in enumerate(methods):
                for mode in MODE_ORDER:
                    row = agg[(agg["method"] == meth) & (agg["mode"] == mode)]
                    if not row.empty:
                        y_data[(meth, mode)] = row["mean"].values[0] + row["ci"].values[0]
            _annotate_sig_brackets(ax, test_results, model, methods, x, width, y_data)

        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=9)
        ax.set_title(model, fontweight="bold")
        ax.set_ylim(bottom=0)
        if ax is axes[0]:
            ax.set_ylabel("Mean Iterations")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=MODE_COLORS[m]) for m in MODE_ORDER
    ]
    fig.legend(
        handles,
        [MODE_LABELS[m] for m in MODE_ORDER],
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.02),
    )
    fig.suptitle("Iterations by Model and Mode", fontsize=14)
    _savefig(fig, outdir, "fig_per_model")
    return fig


def fig_difficulty_vs_savings(df: pd.DataFrame, outdir: str, model_order: list[str]):
    """Scatter: untargeted difficulty vs opportunistic savings."""
    ok = df.copy()
    if ok.empty:
        print("  Skipping fig_difficulty_vs_savings: no successful runs")
        return None
    key_cols = ["model", "method", "epsilon", "seed", "image"]

    unt = ok[ok["mode"] == "untargeted"][key_cols + ["iterations"]].rename(
        columns={"iterations": "iter_unt"}
    )
    opp = ok[ok["mode"] == "opportunistic"][key_cols + ["iterations"]].rename(
        columns={"iterations": "iter_opp"}
    )

    merged = unt.merge(opp, on=key_cols, how="inner")
    if merged.empty:
        print("  Skipping fig_difficulty_vs_savings: no paired successful runs")
        return None
    merged["savings_pct"] = (
        (merged["iter_unt"] - merged["iter_opp"]) / merged["iter_unt"] * 100
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    for method in ["SimBA", "SquareAttack"]:
        for model in model_order:
            sub = merged[(merged["method"] == method) & (merged["model"] == model)]
            if sub.empty:
                continue
            ax.scatter(
                sub["iter_unt"],
                sub["savings_pct"],
                marker=METHOD_MARKERS[method],
                c=("#4878CF" if method == "SimBA" else "#D65F5F"),
                alpha=0.65,
                s=50,
                edgecolors="white",
                linewidths=0.5,
                label=f"{method} / {model}",
            )

    if len(merged) >= 3:
        slope, intercept, r, p, se = stats.linregress(
            merged["iter_unt"], merged["savings_pct"]
        )
        xs = np.linspace(merged["iter_unt"].min(), merged["iter_unt"].max(), 100)
        ax.plot(xs, slope * xs + intercept, "--", color="gray", linewidth=1.5,
                label=f"Trend (r={r:.2f}, p={p:.3f})")

    ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Untargeted Iterations (difficulty)")
    ax.set_ylabel("Opportunistic Savings (%)")
    ax.set_title("Does Opportunistic Targeting Help More on Harder Images?")
    ax.legend(fontsize=8, ncol=2, loc="best")
    _savefig(fig, outdir, "fig_difficulty_vs_savings")
    return fig


def fig_lock_match(df: pd.DataFrame, outdir: str, model_order: list[str]):
    """Lock-match analysis: does opportunistic lock the same class as untargeted?"""
    ok = df.copy()
    if ok.empty:
        print("  Skipping fig_lock_match: no successful runs")
        return None
    key_cols = ["model", "method", "epsilon", "seed", "image"]

    unt = ok[ok["mode"] == "untargeted"][key_cols + ["adversarial_class"]].rename(
        columns={"adversarial_class": "unt_class"}
    )
    opp = ok[ok["mode"] == "opportunistic"][key_cols + ["locked_class"]].copy()
    opp = opp[opp["locked_class"].notna()]

    merged = unt.merge(opp, on=key_cols, how="inner")
    if merged.empty:
        print("  Skipping fig_lock_match: no paired runs with lock data")
        return None
    merged["match"] = merged["unt_class"] == merged["locked_class"]

    match_rate = (
        merged.groupby(["model", "method"])["match"]
        .mean()
        .reset_index()
        .rename(columns={"match": "match_rate"})
    )

    models = [m for m in model_order if m in match_rate["model"].unique()]
    methods = ["SimBA", "SquareAttack"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(models))
    width = 0.3
    for i, method in enumerate(methods):
        sub = match_rate[match_rate["method"] == method].set_index("model").reindex(models)
        color = "#4878CF" if method == "SimBA" else "#D65F5F"
        bars = ax.bar(
            x + (i - 0.5) * width,
            sub["match_rate"] * 100,
            width,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            label=method,
        )
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 1.5,
                    f"{h:.0f}%",
                    ha="center",
                    fontsize=9,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Lock-Match Rate (%)")
    ax.set_title("Does Opportunistic Lock onto the Same Class as Untargeted?")
    ax.set_ylim(0, 115)
    ax.legend()

    n_opp_success = len(df[(df["mode"] == "opportunistic") & df["success"]])
    n_opp_total = len(df[df["mode"] == "opportunistic"])
    ax.text(
        0.5, -0.15,
        f"Opportunistic success rate: {n_opp_success}/{n_opp_total} "
        f"({n_opp_success / n_opp_total * 100:.0f}%) — succeeds even when locking a different class",
        ha="center", transform=ax.transAxes, fontsize=9, style="italic",
    )
    _savefig(fig, outdir, "fig_lock_match")
    return fig


# ===========================================================================
# Progress Metric Figures (robust benchmarks — uses all runs)
# ===========================================================================

def fig_lock_match_robust(df: pd.DataFrame, outdir: str, model_order: list[str]):
    """Lock-match rate for robust models: does OT lock onto the same class
    that untargeted naturally reaches?

    Shows the collapse from standard-level match rates (~80%) to near-zero,
    illustrating the decoy class problem.
    """
    key_cols = ["model", "method", "epsilon", "seed", "image"]

    # Need untargeted adversarial_class (from any run, not just successful)
    # and opportunistic locked_class
    unt = df[df["mode"] == "untargeted"][key_cols + ["adversarial_class"]].rename(
        columns={"adversarial_class": "unt_class"}
    )
    opp = df[df["mode"] == "opportunistic"][key_cols + ["locked_class"]].copy()
    opp = opp[opp["locked_class"].notna()]

    merged = unt.merge(opp, on=key_cols, how="inner")
    if merged.empty:
        print("  Skipping fig_lock_match_robust: no paired runs with lock data")
        return None
    merged["match"] = merged["unt_class"] == merged["locked_class"]

    match_rate = (
        merged.groupby(["model", "method"])["match"]
        .mean()
        .reset_index()
        .rename(columns={"match": "match_rate"})
    )

    models = [m for m in model_order if m in match_rate["model"].unique()]
    methods = ["SimBA", "SquareAttack"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(models))
    width = 0.3
    for i, method in enumerate(methods):
        sub = match_rate[match_rate["method"] == method].set_index("model").reindex(models)
        color = "#4878CF" if method == "SimBA" else "#D65F5F"
        bars = ax.bar(
            x + (i - 0.5) * width,
            sub["match_rate"] * 100,
            width,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            label=method,
        )
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 1.5,
                    f"{h:.0f}%",
                    ha="center",
                    fontsize=9,
                )

    # Reference line for standard benchmark match rates
    ax.axhline(80, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(
        len(models) - 0.5, 82, "Standard benchmark ~80%",
        fontsize=8, color="gray", ha="right",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Lock-Match Rate (%)")
    ax.set_title("Decoy Class Problem: OT Locks onto Wrong Class on Robust Models")
    ax.set_ylim(0, 115)
    ax.legend()
    _savefig(fig, outdir, "fig_lock_match")
    return fig


# ===========================================================================
# Publication Figures (2-mode: untargeted vs opportunistic)
# ===========================================================================

def fig_cdf(df: pd.DataFrame, outdir: str):
    """CDF: cumulative attack success rate vs query budget."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    sub = df[df["mode"].isin(["untargeted", "opportunistic"])].copy()

    q_min, q_max = 10, 10_000
    thresholds = np.unique(np.geomspace(q_min, q_max, 200).astype(int))

    medians = {}

    for method in ["SimBA", "SquareAttack"]:
        for mode in ["untargeted", "opportunistic"]:
            group = sub[(sub["method"] == method) & (sub["mode"] == mode)]
            n_total = len(group)
            if n_total == 0:
                continue

            asr = []
            for q in thresholds:
                n_success = ((group["success"]) & (group["iterations"] <= q)).sum()
                asr.append(n_success / n_total)

            asr = np.array(asr)
            ax.plot(
                thresholds, asr,
                color=_pub_color(mode),
                linestyle=_pub_linestyle(method),
                linewidth=2,
                label=_pub_label(method, mode),
            )

            above_half = np.where(asr >= 0.5)[0]
            if len(above_half):
                medians[(method, mode)] = thresholds[above_half[0]]

    # Annotate median vertical lines & speedup
    anno_idx = 0
    y_positions = [0.52, 0.40]
    for method in ["SimBA", "SquareAttack"]:
        m_unt = medians.get((method, "untargeted"))
        m_opp = medians.get((method, "opportunistic"))
        if m_unt is not None:
            ax.axvline(m_unt, color=COLOR_UNTARGETED, linestyle=":", alpha=0.5, linewidth=1)
        if m_opp is not None:
            ax.axvline(m_opp, color=COLOR_OPPORTUNISTIC, linestyle=":", alpha=0.5, linewidth=1)
        if m_unt and m_opp and m_opp > 0:
            speedup = m_unt / m_opp
            if speedup > 1.05:
                short = "SimBA" if method == "SimBA" else "Square"
                y_pos = y_positions[anno_idx]
                mid_x = np.sqrt(m_unt * m_opp)
                ax.annotate(
                    f"{short}: {speedup:.1f}x",
                    xy=(m_opp, y_pos),
                    xytext=(mid_x, y_pos + 0.08),
                    fontsize=9, fontweight="bold",
                    color=COLOR_OPPORTUNISTIC,
                    arrowprops=dict(arrowstyle="->", color=COLOR_OPPORTUNISTIC, lw=0.8),
                )
                anno_idx += 1

    ax.set_xscale("log")
    ax.set_xlabel("Query Budget")
    ax.set_ylabel("Attack Success Rate")
    ax.set_title("Cumulative Success Rate vs. Query Budget")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(q_min, q_max)
    ax.legend(loc="lower right")
    _savefig(fig, outdir, "fig_cdf")
    return fig


def fig_cdf_per_model(df: pd.DataFrame, outdir: str, model_order: list[str],
                      n_bootstrap: int = 1000):
    """Per-model CDF curves with bootstrap 90% CI bands.

    Layout: len(model_order) rows × 2 columns (one per method).
    Each subplot has 3 curves (untargeted, targeted, opportunistic).
    """
    methods = ["SimBA", "SquareAttack"]
    models = [m for m in model_order if m in df["model"].unique()]
    nrows, ncols = len(models), len(methods)

    budgets = np.arange(50, 10_001, 50)  # 200 points
    rng = np.random.RandomState(0)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.5 * nrows),
                             sharex=True, sharey=True)
    if nrows == 1:
        axes = axes[np.newaxis, :]

    for row, model in enumerate(models):
        for col, method in enumerate(methods):
            ax = axes[row, col]
            subset = df[(df["model"] == model) & (df["method"] == method)]

            for mode in MODE_ORDER:
                mode_df = subset[subset["mode"] == mode]
                image_names = mode_df["image"].unique()
                n_images = len(image_names)
                if n_images == 0:
                    continue

                # Pre-compute per-image success iteration (NaN if failed)
                img_iter = {}
                for name in image_names:
                    r = mode_df[mode_df["image"] == name].iloc[0]
                    img_iter[name] = r["iterations"] if r["success"] else np.nan

                # Bootstrap CDFs
                all_cdfs = np.empty((n_bootstrap, len(budgets)))
                for b in range(n_bootstrap):
                    sample = rng.choice(image_names, size=n_images, replace=True)
                    iters = np.array([img_iter[n] for n in sample])
                    success_iters = np.sort(iters[~np.isnan(iters)])
                    counts = np.searchsorted(success_iters, budgets, side="right")
                    all_cdfs[b] = counts / n_images

                cdf_mean = all_cdfs.mean(axis=0)
                ci_lo = np.percentile(all_cdfs, 5, axis=0)
                ci_hi = np.percentile(all_cdfs, 95, axis=0)

                color = MODE_COLORS[mode]
                ax.plot(budgets, cdf_mean, color=color, linewidth=1.5,
                        label=MODE_LABELS[mode])
                ax.fill_between(budgets, ci_lo, ci_hi, color=color, alpha=0.12)

            ax.set_ylim(-0.02, 1.02)
            ax.set_xlim(budgets[0], budgets[-1])
            if row == nrows - 1:
                ax.set_xlabel("Query Budget")
            if col == 0:
                ax.set_ylabel("Success Rate")
            ax.set_title(f"{model} — {method}", fontsize=11, fontweight="bold")
            if row == 0 and col == ncols - 1:
                ax.legend(loc="lower right", fontsize=8)

    fig.suptitle("Per-Model CDF with Bootstrap 90% CI", fontsize=14, y=1.01)
    _savefig(fig, outdir, "fig_cdf_per_model")
    return fig


def fig_violin(df: pd.DataFrame, outdir: str):
    """Split violin plot of query counts for successful attacks."""
    import seaborn as sns

    ok = df[
        df["mode"].isin(["untargeted", "opportunistic"])
    ].copy()
    if ok.empty:
        print("  Skipping fig_violin: no successful runs")
        return None
    ok["Mode"] = ok["mode"].map({
        "untargeted": "Untargeted",
        "opportunistic": "Opportunistic",
    })
    ok["Method"] = ok["method"]

    fig, ax = plt.subplots(figsize=(6, 5))

    sns.violinplot(
        data=ok,
        x="Method",
        y="iterations",
        hue="Mode",
        split=True,
        inner="quart",
        palette={"Untargeted": COLOR_UNTARGETED, "Opportunistic": COLOR_OPPORTUNISTIC},
        ax=ax,
        linewidth=1,
        cut=0,
    )

    sns.stripplot(
        data=ok,
        x="Method",
        y="iterations",
        hue="Mode",
        dodge=True,
        palette={"Untargeted": COLOR_UNTARGETED, "Opportunistic": COLOR_OPPORTUNISTIC},
        ax=ax,
        size=3,
        alpha=0.3,
        jitter=True,
        legend=False,
    )

    ax.set_yscale("log")
    ax.set_ylabel("Queries to Success (log scale)")
    ax.set_xlabel("")
    ax.set_title("Query Distribution: Untargeted vs. Opportunistic")

    for method in ["SimBA", "SquareAttack"]:
        for mode_val in ["untargeted", "opportunistic"]:
            vals = ok[(ok["Method"] == method) & (ok["mode"] == mode_val)]["iterations"]
            if len(vals):
                med = vals.median()
                x_pos = 0 if method == "SimBA" else 1
                offset = -0.22 if mode_val == "untargeted" else 0.22
                ax.text(
                    x_pos + offset, med,
                    f"  {med:.0f}",
                    ha="center", va="center", fontsize=8, fontweight="bold",
                    color=_pub_color(mode_val),
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8),
                )

    ax.legend(title="", loc="upper right")
    _savefig(fig, outdir, "fig_violin")
    return fig


# ---------------------------------------------------------------------------
# Lock-in dynamics (live attack replay)
# ---------------------------------------------------------------------------
def _replay_attack(method_name, model, x, y_true_tensor, seed, opportunistic, device,
                    source="standard"):
    """Run a single attack and return confidence_history."""
    import torch
    from src.attacks.simba import SimBA
    from src.attacks.square import SquareAttack

    eps = 8 / 255
    max_iter = 10_000
    stability_threshold = STABILITY_THRESHOLDS[source]

    if method_name == "SimBA":
        attack = SimBA(
            model=model, epsilon=eps, max_iterations=max_iter,
            device=device, use_dct=True, pixel_range=(0.0, 1.0),
        )
    else:
        attack = SquareAttack(
            model=model, epsilon=eps, max_iterations=max_iter,
            device=device, loss="ce", normalize=False, seed=seed,
        )

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    attack.generate(
        x, y_true_tensor,
        track_confidence=True,
        targeted=False,
        early_stop=True,
        opportunistic=opportunistic,
        stability_threshold=stability_threshold,
    )
    return attack.confidence_history


def _best_ot_image(csv_path: str, model_name: str, method: str) -> tuple[str, int]:
    """Find the image with the largest OT iteration savings for a given model/method.

    Returns (image_name, seed) for the run with the biggest iter_unt - iter_opp delta.
    """
    df = pd.read_csv(csv_path)
    df["iterations"] = pd.to_numeric(df["iterations"])
    key = ["model", "method", "epsilon", "seed", "image"]
    sub = df[df["model"] == model_name]
    unt = sub[sub["mode"] == "untargeted"][key + ["iterations"]].rename(
        columns={"iterations": "iter_unt"})
    opp = sub[sub["mode"] == "opportunistic"][key + ["iterations"]].rename(
        columns={"iterations": "iter_opp"})
    merged = unt.merge(opp, on=key, how="inner")
    merged = merged[merged["method"] == method]
    merged["delta"] = merged["iter_unt"] - merged["iter_opp"]
    best = merged.loc[merged["delta"].idxmax()]
    return best["image"], int(best["seed"])


def _resolve_image_path(image_name: str, val_dir: str = "data/imagenet/val") -> Path:
    """Resolve an image filename to its full path under the ImageFolder structure."""
    matches = list(Path(val_dir).glob(f"**/{image_name}"))
    if matches:
        return matches[0]
    # Fallback: try data/ directly (old demo images)
    return Path("data") / image_name


def fig_lockin(outdir: str, source: str = "standard", device_str: str = "cuda",
               csv_path: str | None = None):
    """Lock-in dynamics case study: side-by-side SquareAttack & SimBA."""
    import torch
    from benchmark import load_benchmark_model, load_benchmark_image, get_true_label

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    model_name = CASE_MODELS[source]
    if csv_path is None:
        csv_path = f"results/benchmark_{source}.csv"

    # Pick the best showcase images from benchmark data
    cases = []
    for method in ["SimBA", "SquareAttack"]:
        img_name, seed = _best_ot_image(csv_path, model_name, method)
        img_path = _resolve_image_path(img_name)
        short = "Square Attack" if method == "SquareAttack" else method
        cases.append({
            "method": method,
            "image": img_path,
            "seed": seed,
            "title": f"{short} — {model_name} — {img_name}",
        })
        print(f"  Best OT showcase for {method}: {img_name} (seed={seed})")

    print(f"  Loading model for replay ({model_name}, {source}) ...")
    model = load_benchmark_model(model_name, source, device)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, case in zip(axes, cases):
        print(f"  Replaying {case['method']} on {case['image'].name} (seed={case['seed']}) ...")
        x = load_benchmark_image(case["image"], device)
        y_true = get_true_label(model, x)
        y_true_tensor = torch.tensor([y_true], device=device)

        hist_unt = _replay_attack(
            case["method"], model, x, y_true_tensor, case["seed"],
            opportunistic=False, device=device, source=source,
        )
        hist_opp = _replay_attack(
            case["method"], model, x, y_true_tensor, case["seed"],
            opportunistic=True, device=device, source=source,
        )

        # Untargeted traces (faded)
        ax.plot(
            hist_unt["iterations"], hist_unt["original_class"],
            color=COLOR_UNTARGETED, alpha=0.3, linewidth=1.5, linestyle="-",
            label="Original class (untargeted)",
        )
        ax.plot(
            hist_unt["iterations"], hist_unt["max_other_class"],
            color=COLOR_OPPORTUNISTIC, alpha=0.3, linewidth=1.5, linestyle="--",
            label="Max other class (untargeted)",
        )

        # Opportunistic traces (vivid)
        ax.plot(
            hist_opp["iterations"], hist_opp["original_class"],
            color=COLOR_UNTARGETED, alpha=1.0, linewidth=2, linestyle="-",
            label="Original class (opportunistic)",
        )
        ax.plot(
            hist_opp["iterations"], hist_opp["max_other_class"],
            color=COLOR_OPPORTUNISTIC, alpha=1.0, linewidth=2, linestyle="--",
            label="Max other class (opportunistic)",
        )

        # Locked class confidence trace
        locked_class = hist_opp.get("locked_class")
        switch_iter = hist_opp.get("switch_iteration")
        if locked_class is not None:
            top_classes = hist_opp.get("top_classes", [])
            target_conf = hist_opp.get("target_class", [])
            locked_conf = []
            for top_dict in top_classes:
                locked_conf.append(top_dict.get(locked_class))
            locked_conf.extend(target_conf)

            iters_all = hist_opp["iterations"]
            valid = [(it, c) for it, c in zip(iters_all, locked_conf) if c is not None]
            if valid:
                v_iters, v_conf = zip(*valid)
                ax.plot(
                    v_iters, v_conf,
                    color="#2ECC71", linewidth=2, linestyle="-",
                    label=f"Locked class {locked_class}",
                )

        # Vertical lock-in line
        if switch_iter is not None:
            ax.axvline(x=switch_iter, color="#2ECC71", linestyle=":", linewidth=1.5)
            ax.annotate(
                f"Lock-in @ {switch_iter}",
                xy=(switch_iter, 0.92),
                xytext=(switch_iter + 200, 0.95),
                fontsize=8, color="#2ECC71",
                arrowprops=dict(arrowstyle="->", color="#2ECC71", lw=0.8),
            )

        # Mark where each attack ended
        unt_end = hist_unt["iterations"][-1] if hist_unt["iterations"] else None
        opp_end = hist_opp["iterations"][-1] if hist_opp["iterations"] else None
        if unt_end:
            ax.axvline(unt_end, color=COLOR_UNTARGETED, linestyle="--", alpha=0.4, linewidth=1)
            ax.text(unt_end, 0.02, f"Unt: {unt_end}", fontsize=7, color=COLOR_UNTARGETED,
                    ha="right", rotation=90, va="bottom")
        if opp_end:
            ax.axvline(opp_end, color=COLOR_OPPORTUNISTIC, linestyle="--", alpha=0.4, linewidth=1)
            ax.text(opp_end, 0.02, f"Opp: {opp_end}", fontsize=7, color=COLOR_OPPORTUNISTIC,
                    ha="left", rotation=90, va="bottom")

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Confidence")
        ax.set_title(case["title"])
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.3)

    # Shared legend below
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08))

    fig.suptitle("Lock-in Dynamics: Untargeted (faded) vs. Opportunistic (vivid)", fontsize=13)
    _savefig(fig, outdir, "fig_lockin")
    return fig


# ===========================================================================
# Summary Table
# ===========================================================================
def print_summary(df: pd.DataFrame, source: str = "standard"):
    ok = df
    source_label = "Robust Networks" if source == "robust" else "Standard Networks"
    print("\n" + "=" * 80)
    print(f"BENCHMARK SUMMARY — {source_label}")
    print("=" * 80)

    # --- Success rates ---
    print("\n--- Success Rates ---")
    sr = df.groupby(["method", "mode"], observed=True)["success"].mean().reset_index()
    sr["success"] = (sr["success"] * 100).round(1).astype(str) + "%"
    sr_pivot = sr.pivot(index="method", columns="mode", values="success")
    sr_pivot = sr_pivot[[m for m in MODE_ORDER if m in sr_pivot.columns]]
    print(sr_pivot.to_string())

    # --- Iteration statistics ---
    print("\n--- Iteration Statistics (all runs) ---")
    stats_df = (
        ok.groupby(["method", "mode"], observed=True)["iterations"]
        .agg(["mean", "median", "std", "count"])
        .round(1)
        .reset_index()
    )
    stats_df.columns = ["Method", "Mode", "Mean", "Median", "Std", "N"]
    print(stats_df.to_string(index=False))

    # --- Per-model mean iterations ---
    print("\n--- Per-Model Mean Iterations (all runs) ---")
    pm = (
        ok.groupby(["model", "method", "mode"], observed=True)["iterations"]
        .mean()
        .round(0)
        .reset_index()
    )
    pm_pivot = pm.pivot_table(
        index=["model", "method"], columns="mode", values="iterations",
        observed=True,
    )
    pm_pivot = pm_pivot[[m for m in MODE_ORDER if m in pm_pivot.columns]]
    print(pm_pivot.to_string())

    # --- Savings vs untargeted ---
    print("\n--- Savings: Opportunistic vs Untargeted ---")
    key_cols = ["model", "method", "epsilon", "seed", "image"]
    unt = ok[ok["mode"] == "untargeted"][key_cols + ["iterations"]].rename(
        columns={"iterations": "iter_unt"}
    )
    opp = ok[ok["mode"] == "opportunistic"][key_cols + ["iterations"]].rename(
        columns={"iterations": "iter_opp"}
    )
    tgt = ok[ok["mode"] == "targeted"][key_cols + ["iterations"]].rename(
        columns={"iterations": "iter_tgt"}
    )
    merged = unt.merge(opp, on=key_cols, how="inner").merge(tgt, on=key_cols, how="inner")
    if merged.empty:
        print("  No paired successful runs across all three modes")
    else:
        merged["sav_vs_unt"] = (
            (merged["iter_unt"] - merged["iter_opp"]) / merged["iter_unt"] * 100
        )
        merged["overhead_vs_tgt"] = (
            (merged["iter_opp"] - merged["iter_tgt"]) / merged["iter_tgt"] * 100
        )

        savings = merged.groupby(["method"]).agg(
            mean_savings_vs_unt=("sav_vs_unt", "mean"),
            median_savings_vs_unt=("sav_vs_unt", "median"),
            mean_overhead_vs_tgt=("overhead_vs_tgt", "mean"),
            n_triplets=("sav_vs_unt", "count"),
        ).round(1)
        print(savings.to_string())

        print("\n--- Per-Model Savings vs Untargeted (%) ---")
        pm_sav = merged.groupby(["model", "method"])["sav_vs_unt"].agg(
            ["mean", "median"]
        ).round(1)
        print(pm_sav.to_string())

    # --- Switch iteration stats ---
    print("\n--- Opportunistic Switch Statistics ---")
    opp_all = df[df["mode"] == "opportunistic"].copy()
    switched = opp_all[opp_all["switch_iteration"].notna()]
    not_switched = opp_all[opp_all["switch_iteration"].isna()]
    print(f"  Total opportunistic runs: {len(opp_all)}")
    print(f"  Switched: {len(switched)} ({len(switched)/len(opp_all)*100:.1f}%)")
    print(f"  Did not switch: {len(not_switched)} ({len(not_switched)/len(opp_all)*100:.1f}%)")

    if len(switched):
        sw = switched.groupby("method")["switch_iteration"].agg(
            ["mean", "median", "min", "max"]
        ).round(1)
        print(sw.to_string())

    # --- Lock-match rate ---
    print("\n--- Lock-Match Rate (opportunistic locked_class == untargeted adversarial_class) ---")
    unt2 = ok[ok["mode"] == "untargeted"][key_cols + ["adversarial_class"]].rename(
        columns={"adversarial_class": "unt_class"}
    )
    opp2 = ok[ok["mode"] == "opportunistic"][key_cols + ["locked_class"]]
    opp2 = opp2[opp2["locked_class"].notna()]
    lm = unt2.merge(opp2, on=key_cols, how="inner")
    if lm.empty:
        print("  No paired runs with lock data")
    else:
        lm["match"] = lm["unt_class"] == lm["locked_class"]
        lm_rate = lm.groupby(["method"])["match"].mean() * 100
        print(lm_rate.round(1).to_string())

    print("\n" + "=" * 80)


def _sig_stars(p):
    """Significance stars from p-value."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def compute_paired_tests(df: pd.DataFrame, metric: str = "iterations",
                         filter_success: bool = True,
                         pool_models: bool = False) -> pd.DataFrame:
    """Compute paired Wilcoxon + t-tests for all 3 mode pairs.

    Args:
        df: Benchmark DataFrame.
        metric: Column to compare ("iterations" or "margin_final").
        filter_success: If True, only include images where both modes succeeded.
        pool_models: If True, pool all models into one test per method × pair.

    Returns:
        DataFrame with one row per (model, method, modeA, modeB) test.
    """
    work = df.copy()
    if metric == "margin_final" and "margin_final" not in work.columns:
        work["margin_final"] = 1.0 - work["confusion_final"]

    key_cols = ["model", "method", "epsilon", "seed", "image"]
    pairs = [
        ("untargeted", "opportunistic"),
        ("untargeted", "targeted"),
        ("opportunistic", "targeted"),
    ]

    if pool_models:
        model_groups = ["_pooled_"]
    else:
        model_groups = list(work["model"].unique())

    rows = []
    for model in model_groups:
        for method in work["method"].unique():
            for modeA, modeB in pairs:
                if pool_models:
                    a = work[(work["method"] == method) & (work["mode"] == modeA)]
                    b = work[(work["method"] == method) & (work["mode"] == modeB)]
                else:
                    a = work[(work["model"] == model) & (work["method"] == method)
                             & (work["mode"] == modeA)]
                    b = work[(work["model"] == model) & (work["method"] == method)
                             & (work["mode"] == modeB)]
                if a.empty or b.empty:
                    continue

                a_df = a[key_cols + [metric, "success"]].rename(
                    columns={metric: "val_a", "success": "suc_a"})
                b_df = b[key_cols + [metric, "success"]].rename(
                    columns={metric: "val_b", "success": "suc_b"})
                merged = a_df.merge(b_df, on=key_cols, how="inner")

                if filter_success:
                    merged = merged[merged["suc_a"] & merged["suc_b"]]

                if len(merged) < 2:
                    continue

                va = merged["val_a"].values
                vb = merged["val_b"].values
                med_a = np.median(va)
                med_b = np.median(vb)
                savings = (med_a - med_b) / med_a * 100 if med_a != 0 else 0.0

                try:
                    w_stat, w_p = stats.wilcoxon(va, vb, alternative="two-sided")
                except ValueError:
                    w_stat, w_p = 0.0, 1.0

                t_stat, t_p = stats.ttest_rel(va, vb)

                rows.append({
                    "model": model, "method": method,
                    "modeA": modeA, "modeB": modeB,
                    "N": len(merged),
                    "medA": med_a, "medB": med_b, "savings_pct": savings,
                    "w_stat": w_stat, "w_p": w_p,
                    "t_stat": t_stat, "t_p": t_p,
                })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)

    # Bonferroni correction
    n_tests = len(result)
    result["w_p_bonf"] = (result["w_p"] * n_tests).clip(upper=1.0)
    result["t_p_bonf"] = (result["t_p"] * n_tests).clip(upper=1.0)
    result["sig"] = result["w_p_bonf"].apply(_sig_stars)

    return result


def print_paired_tests(df: pd.DataFrame, source: str = "standard"):
    """Print formatted tables of all paired statistical tests."""

    # --- Iteration tests ---
    test_iter = compute_paired_tests(df, metric="iterations", filter_success=True)
    if not test_iter.empty:
        label = "Paired Tests: Iterations (successful runs, Bonferroni-corrected)"
        print(f"\n{label}")
        header = (f"{'Model':<22} {'Method':<14} {'Pair':<20} {'N':>4}  "
                  f"{'Med A':>7} {'Med B':>7} {'Sav%':>7}  "
                  f"{'W':>8} {'p(W)':>9} {'p(W)adj':>8} {'Sig':>4}  "
                  f"{'t':>7} {'p(t)':>9} {'p(t)adj':>8}")
        print(header)
        print("-" * len(header))
        for _, r in test_iter.iterrows():
            pair_label = f"{r['modeA'][:3]}v{r['modeB'][:3]}"
            print(f"{r['model']:<22} {r['method']:<14} {pair_label:<20} "
                  f"{r['N']:>4}  "
                  f"{r['medA']:>7.0f} {r['medB']:>7.0f} {r['savings_pct']:>+6.1f}%  "
                  f"{r['w_stat']:>8.0f} {r['w_p']:>9.4g} {r['w_p_bonf']:>8.4g} "
                  f"{r['sig']:>4}  "
                  f"{r['t_stat']:>7.2f} {r['t_p']:>9.4g} {r['t_p_bonf']:>8.4g}")

    return test_iter


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Analyze benchmark results and generate all figures."
    )
    parser.add_argument(
        "--source", choices=["standard", "robust"], default="standard",
        help="Model source (default: standard)",
    )
    parser.add_argument(
        "--csv", default=None,
        help="Path to benchmark CSV (default: results/benchmark_{source}.csv)",
    )
    parser.add_argument(
        "--outdir", default=None,
        help="Output directory for figures (default: results/figures/{source})",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display figures interactively",
    )
    parser.add_argument(
        "--skip-replay", action="store_true",
        help="Skip lock-in dynamics figure (no model loading required)",
    )
    args = parser.parse_args()

    if args.csv is None:
        args.csv = f"results/benchmark_{args.source}.csv"
    if args.outdir is None:
        args.outdir = f"results/figures/{args.source}"

    source = args.source
    model_order = MODEL_ORDERS[source]

    if not args.show:
        matplotlib.use("Agg")

    _setup_style()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Loading data from {args.csv} ...")
    df = load_data(args.csv)
    print(f"  {len(df)} rows, {df['model'].nunique()} models, "
          f"{df['method'].nunique()} methods, {df['mode'].nunique()} modes")

    print(f"\n  Source: {source}, models: {model_order}")

    # --- Compute statistical tests (used by figures + printed tables) ---
    print("\n=== Statistical Tests ===")
    test_iter = compute_paired_tests(df, metric="iterations", filter_success=True)

    # --- Diagnostic Figures (all sources) ---
    print("\n=== Diagnostic Figures ===")
    print("\n--- Headline Bars ---")
    fig_headline_bars(df, args.outdir, test_results=test_iter)
    print("\n--- Per-Model Breakdown ---")
    fig_per_model(df, args.outdir, model_order, test_results=test_iter)
    print("\n--- Difficulty vs Savings ---")
    fig_difficulty_vs_savings(df, args.outdir, model_order)
    print("\n--- Lock-Match Analysis ---")
    fig_lock_match(df, args.outdir, model_order)

    # --- Publication Figures (all sources) ---
    print("\n=== Publication Figures ===")
    print("\n--- CDF ---")
    fig_cdf(df, args.outdir)
    print("\n--- Violin ---")
    fig_violin(df, args.outdir)

    if not args.skip_replay:
        print("\n--- Lock-in Dynamics (live replay) ---")
        fig_lockin(args.outdir, source=source, csv_path=args.csv)
    else:
        print("\n--- Lock-in Dynamics: Skipped (--skip-replay) ---")

    print("\n--- Per-Model CDF (bootstrap CI) ---")
    fig_cdf_per_model(df, args.outdir, model_order)

    # --- Robust-only: lock-match decoy analysis ---
    if source == "robust":
        print("\n--- Lock-Match Robust (decoy analysis) ---")
        fig_lock_match_robust(df, args.outdir, model_order)

    print("\n--- Paired Statistical Tests ---")
    print_paired_tests(df, source=source)

    print_summary(df, source=source)

    if args.show:
        plt.show()

    print(f"\nAll figures saved to {args.outdir}/")


if __name__ == "__main__":
    main()
