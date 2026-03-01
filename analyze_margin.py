"""Margin-loss ablation analysis (Issue #12).

Compares 4 SquareAttack configurations on ResNet-50 (100 images, 15K budget):
  1. Margin untargeted        (from benchmark_margin.csv)
  2. Margin + OT              (from benchmark_margin.csv)
  3. CE + OT                  (from benchmark_winrate.csv)
  4. CE oracle targeted       (from benchmark_winrate.csv)

Produces:
  - Console summary table (mean/median iters, success rate)
  - Wilcoxon signed-rank test: margin untargeted vs margin + OT
  - CDF figure: 4 curves with bootstrap 90% CI
  - Grouped bar chart: mean iterations per configuration

Usage:
    python analyze_margin.py                       # Default (no display)
    python analyze_margin.py --show                # Interactive display
    python analyze_margin.py --outdir results/figures_margin
"""

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


# ===========================================================================
# Style (matches analyze_benchmark.py)
# ===========================================================================
def _setup_style():
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
    try:
        from matplotlib.texmanager import TexManager
        TexManager._run_checked_subprocess(["latex", "--version"], "latex")
        plt.rcParams["text.usetex"] = True
    except Exception:
        plt.rcParams["text.usetex"] = False


def _savefig(fig, outdir: str, name: str):
    fig.savefig(os.path.join(outdir, f"{name}.png"), bbox_inches="tight")
    fig.savefig(os.path.join(outdir, f"{name}.pdf"), bbox_inches="tight")
    print(f"  Saved {name}.png / .pdf")


# ===========================================================================
# Configuration labels and colors
# ===========================================================================
CONFIG_ORDER = [
    "margin_untargeted",
    "margin_ot",
    "ce_ot",
    "ce_oracle",
]
CONFIG_LABELS = {
    "margin_untargeted": "Margin untargeted",
    "margin_ot": "Margin + OT",
    "ce_ot": "CE + OT",
    "ce_oracle": "CE oracle",
}
CONFIG_COLORS = {
    "margin_untargeted": "#4878CF",   # blue
    "margin_ot": "#6BA353",           # green
    "ce_ot": "#D65F5F",              # red
    "ce_oracle": "#E8873A",          # orange
}
CONFIG_LINESTYLES = {
    "margin_untargeted": "-",
    "margin_ot": "-",
    "ce_ot": "--",
    "ce_oracle": "--",
}


# ===========================================================================
# Data loading
# ===========================================================================
def load_margin_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["iterations"] = pd.to_numeric(df["iterations"])
    df["success"] = df["success"].map(
        {"True": True, "False": False, True: True, False: False}
    )
    return df


def load_winrate_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["iterations"] = pd.to_numeric(df["iterations"])
    df["success"] = df["success"].map(
        {"True": True, "False": False, True: True, False: False}
    )
    return df


def build_configs(margin_df: pd.DataFrame,
                  winrate_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Extract the 4 configurations as separate DataFrames keyed by image."""
    sq_win = winrate_df[winrate_df["method"] == "SquareAttack"]

    configs = {
        "margin_untargeted": margin_df[margin_df["mode"] == "untargeted"].copy(),
        "margin_ot": margin_df[margin_df["mode"] == "opportunistic"].copy(),
        "ce_ot": sq_win[sq_win["mode"] == "opportunistic"].copy(),
        "ce_oracle": sq_win[sq_win["mode"] == "targeted"].copy(),
    }
    # Normalize: set image as index for easy pairing
    for key in configs:
        configs[key] = configs[key].set_index("image")
    return configs


# ===========================================================================
# Console summary
# ===========================================================================
def print_summary(configs: dict[str, pd.DataFrame]):
    print(f"\n{'='*70}")
    print("  Margin-Loss Ablation Summary (SquareAttack, ResNet-50, 15K budget)")
    print(f"{'='*70}")

    header = f"  {'Configuration':<22s} {'N':>4s} {'Success':>8s} {'Mean':>8s} {'Median':>8s} {'Std':>8s}"
    print(header)
    print(f"  {'-'*60}")

    for key in CONFIG_ORDER:
        df = configs[key]
        n = len(df)
        sr = df["success"].mean()
        succ = df[df["success"]]
        mean_i = succ["iterations"].mean() if len(succ) > 0 else float("nan")
        med_i = succ["iterations"].median() if len(succ) > 0 else float("nan")
        std_i = succ["iterations"].std() if len(succ) > 1 else float("nan")
        print(f"  {CONFIG_LABELS[key]:<22s} {n:>4d} {sr:>7.1%} {mean_i:>8.0f} {med_i:>8.0f} {std_i:>8.0f}")


def print_paired_test(configs: dict[str, pd.DataFrame]):
    """Wilcoxon signed-rank: margin untargeted vs margin + OT."""
    mu = configs["margin_untargeted"]
    mo = configs["margin_ot"]

    # Find images where both succeed
    common = mu.index.intersection(mo.index)
    both_succeed = [img for img in common
                    if mu.loc[img, "success"] and mo.loc[img, "success"]]
    n = len(both_succeed)

    print(f"\n  Paired test: Margin untargeted vs Margin + OT")
    print(f"  Both-succeed images: {n}")

    if n < 10:
        print(f"  Insufficient paired successes for Wilcoxon test (need >= 10)")
        return

    iters_u = np.array([mu.loc[img, "iterations"] for img in both_succeed])
    iters_o = np.array([mo.loc[img, "iterations"] for img in both_succeed])
    diff = iters_u - iters_o

    print(f"  Mean iterations: untargeted={iters_u.mean():.0f}, "
          f"OT={iters_o.mean():.0f}")
    print(f"  Mean savings: {diff.mean():+.0f} iters "
          f"({diff.mean() / iters_u.mean():+.1%})")

    # Wilcoxon test (two-sided)
    if np.all(diff == 0):
        print(f"  All differences are zero — modes are identical.")
        return

    stat, p = wilcoxon(iters_u, iters_o)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"  Wilcoxon: W={stat:.0f}, p={p:.2e} {sig}")


# ===========================================================================
# CDF computation (adapted from analyze_winrate.py)
# ===========================================================================
BUDGET_CAP = 15_000


def bootstrap_cdf(df: pd.DataFrame, budgets: np.ndarray,
                  n_bootstrap: int = 1000, seed: int = 0):
    """Bootstrap CDF with 90% confidence intervals."""
    rng = np.random.RandomState(seed)
    image_names = df.index.unique()
    n_images = len(image_names)
    if n_images == 0:
        z = np.zeros(len(budgets))
        return z, z.copy(), z.copy()

    img_iter = {}
    for name in image_names:
        row = df.loc[name]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        if row["success"] and row["iterations"] <= BUDGET_CAP:
            img_iter[name] = row["iterations"]
        else:
            img_iter[name] = np.nan

    all_cdfs = np.empty((n_bootstrap, len(budgets)))
    for b in range(n_bootstrap):
        sample = rng.choice(image_names, size=n_images, replace=True)
        iters = np.array([img_iter[n] for n in sample])
        success_iters = np.sort(iters[~np.isnan(iters)])
        counts = np.searchsorted(success_iters, budgets, side="right")
        all_cdfs[b] = counts / n_images

    return all_cdfs.mean(axis=0), np.percentile(all_cdfs, 5, axis=0), \
        np.percentile(all_cdfs, 95, axis=0)


# ===========================================================================
# Figures
# ===========================================================================
def fig_margin_cdf(configs: dict[str, pd.DataFrame], outdir: str):
    """CDF plot: 4 configurations overlaid."""
    step = 50
    budgets = np.arange(step, BUDGET_CAP + 1, step)

    fig, ax = plt.subplots(figsize=(8, 5))

    for key in CONFIG_ORDER:
        df = configs[key]
        mean, lo, hi = bootstrap_cdf(df, budgets)
        color = CONFIG_COLORS[key]
        ls = CONFIG_LINESTYLES[key]
        ax.plot(budgets, mean, color=color, linestyle=ls,
                linewidth=1.5, label=CONFIG_LABELS[key])
        ax.fill_between(budgets, lo, hi, color=color, alpha=0.10)

    ax.set_xlabel("Query budget")
    ax.set_ylabel("Success rate (CDF)")
    ax.set_xlim(0, budgets[-1])
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_title("Success Rate vs Query Budget — Loss Ablation\n"
                 "(Square Attack, ResNet-50, 100 images)")
    _savefig(fig, outdir, "fig_margin_cdf")
    return fig


def fig_margin_bars(configs: dict[str, pd.DataFrame], outdir: str):
    """Grouped bar chart: mean iterations (successful runs only)."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    labels, means, sems = [], [], []
    for key in CONFIG_ORDER:
        df = configs[key]
        succ = df[df["success"]]
        labels.append(CONFIG_LABELS[key])
        means.append(succ["iterations"].mean() if len(succ) > 0 else 0)
        sems.append(succ["iterations"].sem() if len(succ) > 1 else 0)

    x = np.arange(len(labels))
    colors = [CONFIG_COLORS[k] for k in CONFIG_ORDER]
    bars = ax.bar(x, means, yerr=sems, color=colors, edgecolor="white",
                  linewidth=0.5, capsize=4)

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{m:.0f}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Mean Iterations (successful runs)")
    ax.set_title("Mean Iterations by Configuration\n"
                 "(Square Attack, ResNet-50, 15K budget)")
    _savefig(fig, outdir, "fig_margin_bars")
    return fig


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Margin-loss ablation analysis"
    )
    parser.add_argument("--margin-csv",
                        default="results/benchmark_margin.csv")
    parser.add_argument("--winrate-csv",
                        default="results/benchmark_winrate.csv")
    parser.add_argument("--outdir", default="results/figures_margin")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if not args.show:
        matplotlib.use("Agg")

    _setup_style()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Loading margin data from {args.margin_csv}")
    margin_df = load_margin_csv(args.margin_csv)
    print(f"  {len(margin_df)} rows")

    print(f"Loading CE data from {args.winrate_csv}")
    winrate_df = load_winrate_csv(args.winrate_csv)
    print(f"  {len(winrate_df)} rows (filtering to SquareAttack)")

    configs = build_configs(margin_df, winrate_df)

    # Console summary
    print_summary(configs)
    print_paired_test(configs)

    # Figures
    print("\nGenerating figures...")
    fig_margin_cdf(configs, args.outdir)
    fig_margin_bars(configs, args.outdir)

    if args.show:
        plt.show()

    print("\nDone.")


if __name__ == "__main__":
    main()
