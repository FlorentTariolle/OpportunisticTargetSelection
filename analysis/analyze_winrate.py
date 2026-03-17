"""CDF analysis: success rate vs query budget figures.

Reads benchmark_winrate.csv and produces one CDF plot per method:
  - fig_winrate_simba: 3 curves (untargeted, targeted, opportunistic)
  - fig_winrate_squareattack: 3 curves (untargeted, targeted, opportunistic)

Both use true CDF: CDF(alpha) = fraction of attacks with t_a <= alpha,
derived from a single run per (method, image, mode) at a fixed budget.
Legacy SimBA rows with budget > 15K are capped at 15K.

Usage:
    python analyze_winrate.py                       # Default CSV + output
    python analyze_winrate.py --show                # Interactive display
    python analyze_winrate.py --csv results/custom.csv --outdir results/figs
"""

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ===========================================================================
# Style configuration (matches analyze_benchmark.py)
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

    try:
        from matplotlib.texmanager import TexManager
        TexManager._run_checked_subprocess(
            ["latex", "--version"], "latex")
        plt.rcParams["text.usetex"] = True
    except Exception:
        plt.rcParams["text.usetex"] = False


def _savefig(fig, outdir: str, name: str):
    fig.savefig(os.path.join(outdir, f"{name}.png"), bbox_inches="tight")
    fig.savefig(os.path.join(outdir, f"{name}.pdf"), bbox_inches="tight")
    print(f"  Saved {name}.png / .pdf")


# ===========================================================================
# Color / style constants (matches analyze_benchmark.py)
# ===========================================================================
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
METHOD_LABELS = {
    "SimBA": "SimBA",
    "SquareAttack": "Square Attack",
}


# ===========================================================================
# Data loading
# ===========================================================================
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["iterations"] = pd.to_numeric(df["iterations"])
    df["budget"] = pd.to_numeric(df["budget"])
    df["success"] = df["success"].map(
        {"True": True, "False": False, True: True, False: False}
    )
    df["switch_iteration"] = pd.to_numeric(
        df["switch_iteration"], errors="coerce"
    )
    df["locked_class"] = pd.to_numeric(df["locked_class"], errors="coerce")
    df["oracle_target"] = pd.to_numeric(df["oracle_target"], errors="coerce")
    df["adversarial_class"] = pd.to_numeric(
        df["adversarial_class"], errors="coerce"
    )
    return df


# ===========================================================================
# CDF computation
# ===========================================================================
BUDGET_CAP = 15_000


def bootstrap_cdf(df: pd.DataFrame, budgets: np.ndarray,
                   n_bootstrap: int = 1000, seed: int = 0):
    """Bootstrap CDF with 90% confidence intervals.

    Args:
        df: DataFrame filtered to a single method (excluding oracle_probe).
        budgets: Array of budget thresholds.
        n_bootstrap: Number of bootstrap samples.
        seed: RNG seed for reproducibility.

    Returns:
        dict mapping mode -> (cdf_mean, ci_lo, ci_hi) arrays.
    """
    rng = np.random.RandomState(seed)
    result = {}
    for mode in MODE_ORDER:
        subset = df[df["mode"] == mode].copy()
        image_names = subset["image"].unique()
        n_images = len(image_names)
        if n_images == 0:
            z = np.zeros(len(budgets))
            result[mode] = (z, z.copy(), z.copy())
            continue

        # Pre-compute per-image success iteration (or NaN if failed/capped)
        img_iter = {}
        for name in image_names:
            row = subset[subset["image"] == name].iloc[0]
            if row["success"] and row["iterations"] <= BUDGET_CAP:
                img_iter[name] = row["iterations"]
            else:
                img_iter[name] = np.nan

        all_cdfs = np.empty((n_bootstrap, len(budgets)))
        for b in range(n_bootstrap):
            sample_names = rng.choice(image_names, size=n_images, replace=True)
            iters = np.array([img_iter[n] for n in sample_names])
            success_iters = np.sort(iters[~np.isnan(iters)])
            counts = np.searchsorted(success_iters, budgets, side="right")
            all_cdfs[b] = counts / n_images

        cdf_mean = all_cdfs.mean(axis=0)
        ci_lo = np.percentile(all_cdfs, 5, axis=0)
        ci_hi = np.percentile(all_cdfs, 95, axis=0)
        result[mode] = (cdf_mean, ci_lo, ci_hi)
    return result


# ===========================================================================
# Figures
# ===========================================================================
def fig_winrate_method(cdf, budgets, method, outdir, show):
    """Single-method CDF plot: 3 curves (one per mode) with 90% CI bands."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for mode in MODE_ORDER:
        color = MODE_COLORS[mode]
        mean, lo, hi = cdf[mode]
        ax.plot(budgets, mean, color=color, linestyle="-",
                linewidth=1.5, label=MODE_LABELS[mode])
        ax.fill_between(budgets, lo, hi, color=color, alpha=0.12)

    ax.set_xlabel("Query budget")
    ax.set_ylabel("Success rate (CDF)")
    ax.set_xlim(0, budgets[-1])
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_title(f"Success Rate vs Query Budget — {METHOD_LABELS[method]} (ResNet-50)")

    name = f"fig_winrate_{method.lower()}"
    _savefig(fig, outdir, name)
    if show:
        plt.show()
    plt.close(fig)


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Analyze winrate benchmark results"
    )
    parser.add_argument("--csv", default="results/benchmark_winrate.csv",
                        help="Path to benchmark CSV")
    parser.add_argument("--outdir", default="results/figures_winrate",
                        help="Output directory for figures")
    parser.add_argument("--show", action="store_true",
                        help="Show interactive plots")
    args = parser.parse_args()

    _setup_style()

    print(f"Loading data from {args.csv}")
    df = load_data(args.csv)

    os.makedirs(args.outdir, exist_ok=True)

    # Unified budget axis: 50-step from 50 to BUDGET_CAP
    step = 50
    budgets = np.arange(step, BUDGET_CAP + 1, step)

    print(f"Budget axis: {budgets[0]}..{budgets[-1]} (step={step}, "
          f"{len(budgets)} points)")

    # Process each method independently
    for method in ["SimBA", "SquareAttack"]:
        df_method = df[(df["method"] == method) & (df["mode"] != "oracle_probe")]
        n_rows = len(df_method)
        n_images = df_method["image"].nunique()
        if n_rows == 0:
            print(f"\n{method}: no data, skipping")
            continue

        print(f"\n{method}: {n_rows} rows, {n_images} images")

        cdf = bootstrap_cdf(df_method, budgets)

        for mode in MODE_ORDER:
            final = cdf[mode][0][-1]
            print(f"  {mode:>14s}: {final:.1%} (at {budgets[-1]})")

        fig_winrate_method(cdf, budgets, method, args.outdir, args.show)

    print("\nDone.")


if __name__ == "__main__":
    main()
