"""Analyze stability-threshold ablation results.

Reads benchmark_ablation_s.csv and produces a 2x2 figure:
  - Top row: SimBA (success rate vs S, mean iterations vs S)
  - Bottom row: SquareAttack (same)

Backwards-compatible: if only one method is present, produces 1x2 figure.

Usage:
    python analyze_ablation_s.py                       # Default CSV + output
    python analyze_ablation_s.py --show                # Interactive display
"""

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ===========================================================================
# Style (matches analyze_winrate.py)
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
# Per-method statistics
# ===========================================================================
METHOD_COLORS = {
    "SimBA": "#2176AE",        # blue
    "SquareAttack": "#E07A30", # orange
}

METHOD_LABELS = {
    "SimBA": "SimBA",
    "SquareAttack": "Square Attack (CE)",
}


def _compute_stats(df_method, s_values):
    """Compute success rate and mean iterations per S value."""
    success_rates = []
    mean_iters = []
    for s in s_values:
        subset = df_method[df_method["s_value"] == s]
        sr = subset["success"].mean() if len(subset) > 0 else 0.0
        success_rates.append(sr)
        succ_subset = subset[subset["success"]]
        avg = succ_subset["iterations"].mean() if len(succ_subset) > 0 else np.nan
        mean_iters.append(avg)
    return success_rates, mean_iters


def _find_best_s(s_values, success_rates, mean_iters):
    """Highest success rate, break ties by lowest mean iterations."""
    return s_values[max(
        range(len(s_values)),
        key=lambda i: (
            success_rates[i],
            -mean_iters[i] if not np.isnan(mean_iters[i]) else float('inf'),
        ),
    )]


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Analyze stability-threshold ablation results"
    )
    parser.add_argument("--csv", default="results/benchmark_ablation_s.csv",
                        help="Path to ablation CSV")
    parser.add_argument("--outdir", default="results/figures_ablation_s",
                        help="Output directory for figures")
    parser.add_argument("--show", action="store_true",
                        help="Show interactive plots")
    args = parser.parse_args()

    _setup_style()

    print(f"Loading data from {args.csv}")
    df = pd.read_csv(args.csv)
    df["iterations"] = pd.to_numeric(df["iterations"])
    df["success"] = df["success"].map(
        {"True": True, "False": False, True: True, False: False}
    )
    df["s_value"] = pd.to_numeric(df["s_value"])

    # Backwards compat: old CSV without 'method' column → assume SimBA
    if "method" not in df.columns:
        df["method"] = "SimBA"

    os.makedirs(args.outdir, exist_ok=True)

    methods = sorted(df["method"].unique())
    n_methods = len(methods)

    print(f"Methods found: {methods}")

    # ---- Compute stats per method (each may have different S values) ----
    stats = {}  # method -> (s_values, success_rates, mean_iters, best_s)
    for method in methods:
        df_m = df[df["method"] == method]
        s_values = sorted(df_m["s_value"].unique())
        sr, mi = _compute_stats(df_m, s_values)
        best_s = _find_best_s(s_values, sr, mi)
        stats[method] = (s_values, sr, mi, best_s)

        label = METHOD_LABELS.get(method, method)
        print(f"\n{label} (S values: {s_values}):")
        for i, s in enumerate(s_values):
            subset = df_m[df_m["s_value"] == s]
            n = len(subset)
            n_succ = int(subset["success"].sum())
            avg_str = f"{mi[i]:.0f}" if not np.isnan(mi[i]) else "N/A"
            print(f"  S={s:>2d}: {sr[i]:.1%} success ({n_succ}/{n}), "
                  f"mean iters={avg_str}")
        print(f"  Optimal S: {best_s}")

    # ---- Figure: n_methods rows x 2 cols ----
    fig, axes = plt.subplots(n_methods, 2,
                             figsize=(10, 4.5 * n_methods),
                             squeeze=False)

    for row, method in enumerate(methods):
        sv, sr, mi, best_s = stats[method]
        color = METHOD_COLORS.get(method, "#6BA353")
        label = METHOD_LABELS.get(method, method)

        ax_sr = axes[row, 0]
        ax_mi = axes[row, 1]

        # Success rate
        ax_sr.plot(sv, sr, 'o-', color=color, linewidth=1.5,
                   markersize=6)
        ax_sr.axvline(best_s, color='gray', linestyle=':', alpha=0.6)
        ax_sr.set_xlabel("Stability threshold $S$")
        ax_sr.set_ylabel("Success rate")
        ax_sr.set_xticks(sv)
        ax_sr.set_ylim(-0.02, 1.02)
        ax_sr.set_title(f"{label} — Success Rate vs $S$")

        # Mean iterations
        ax_mi.plot(sv, mi, 's-', color=color, linewidth=1.5,
                   markersize=6)
        ax_mi.axvline(best_s, color='gray', linestyle=':', alpha=0.6)
        ax_mi.set_xlabel("Stability threshold $S$")
        ax_mi.set_ylabel("Mean iterations (successful)")
        ax_mi.set_xticks(sv)
        ax_mi.set_title(f"{label} — Mean Iterations vs $S$")

    _savefig(fig, args.outdir, "fig_ablation_s")
    if args.show:
        plt.show()
    plt.close(fig)

    print("\nDone.")


if __name__ == "__main__":
    main()
