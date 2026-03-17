"""Analyze naive fixed-iteration switching ablation results.

Reads benchmark_ablation_naive.csv and produces a 2x2 figure:
  - Top row: SimBA (success rate vs T, mean iterations vs T)
  - Bottom row: SquareAttack (same)

OT baseline (stability heuristic) is shown as a horizontal reference line.

Usage:
    python analyze_ablation_naive.py                       # Default CSV + output
    python analyze_ablation_naive.py --show                # Interactive display
"""

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ===========================================================================
# Style (matches analyze_ablation_s.py)
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
    "SimBA": "#2176AE",
    "SquareAttack": "#E07A30",
}

METHOD_LABELS = {
    "SimBA": "SimBA",
    "SquareAttack": "Square Attack (CE)",
}


def _compute_stats(df_method, t_values):
    """Compute success rate and mean iterations per T value."""
    success_rates = []
    mean_iters = []
    for t in t_values:
        subset = df_method[df_method["t_value"] == t]
        sr = subset["success"].mean() if len(subset) > 0 else 0.0
        success_rates.append(sr)
        succ_subset = subset[subset["success"]]
        avg = succ_subset["iterations"].mean() if len(succ_subset) > 0 else np.nan
        mean_iters.append(avg)
    return success_rates, mean_iters


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Analyze naive fixed-iteration switching ablation results"
    )
    parser.add_argument("--csv", default="results/benchmark_ablation_naive.csv",
                        help="Path to ablation CSV")
    parser.add_argument("--outdir", default="results/figures/standard",
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

    os.makedirs(args.outdir, exist_ok=True)

    methods = sorted(df["method"].unique())
    n_methods = len(methods)

    print(f"Methods found: {methods}")

    # Separate OT baseline and naive T rows
    df_ot = df[df["t_value"] == "OT"]
    df_naive = df[df["t_value"] != "OT"].copy()
    df_naive["t_value"] = pd.to_numeric(df_naive["t_value"])

    # ---- Compute stats per method ----
    stats = {}  # method -> (t_values, sr, mi, ot_sr, ot_mi)
    for method in methods:
        df_m = df_naive[df_naive["method"] == method]
        t_values = sorted(df_m["t_value"].unique())
        sr, mi = _compute_stats(df_m, t_values)

        # OT baseline stats
        df_ot_m = df_ot[df_ot["method"] == method]
        ot_sr = df_ot_m["success"].mean() if len(df_ot_m) > 0 else np.nan
        ot_succ = df_ot_m[df_ot_m["success"]]
        ot_mi = ot_succ["iterations"].mean() if len(ot_succ) > 0 else np.nan

        stats[method] = (t_values, sr, mi, ot_sr, ot_mi)

        label = METHOD_LABELS.get(method, method)
        print(f"\n{label} (T values: {t_values}):")
        for i, t in enumerate(t_values):
            subset = df_m[df_m["t_value"] == t]
            n = len(subset)
            n_succ = int(subset["success"].sum())
            avg_str = f"{mi[i]:.0f}" if not np.isnan(mi[i]) else "N/A"
            print(f"  T={t:>3d}: {sr[i]:.1%} success ({n_succ}/{n}), "
                  f"mean iters={avg_str}")
        ot_sr_str = f"{ot_sr:.1%}" if not np.isnan(ot_sr) else "N/A"
        ot_mi_str = f"{ot_mi:.0f}" if not np.isnan(ot_mi) else "N/A"
        print(f"  OT baseline: {ot_sr_str} success, mean iters={ot_mi_str}")

    # ---- Figure: n_methods rows x 2 cols ----
    fig, axes = plt.subplots(n_methods, 2,
                             figsize=(10, 4.5 * n_methods),
                             squeeze=False)

    for row, method in enumerate(methods):
        tv, sr, mi, ot_sr, ot_mi = stats[method]
        color = METHOD_COLORS.get(method, "#6BA353")
        label = METHOD_LABELS.get(method, method)

        ax_sr = axes[row, 0]
        ax_mi = axes[row, 1]

        # Success rate vs T
        ax_sr.plot(tv, sr, 'o-', color=color, linewidth=1.5,
                   markersize=6, label=f"Naive (fixed $T$)")
        if not np.isnan(ot_sr):
            ax_sr.axhline(ot_sr, color='gray', linestyle='--', alpha=0.7,
                          label=f"OT (stability)")
        ax_sr.set_xlabel("Switch iteration $T$")
        ax_sr.set_ylabel("Success rate")
        ax_sr.set_xscale('log')
        ax_sr.set_xticks(tv)
        ax_sr.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax_sr.set_ylim(-0.02, 1.02)
        ax_sr.set_title(f"{label} — Success Rate vs $T$")
        ax_sr.legend()

        # Mean iterations vs T
        ax_mi.plot(tv, mi, 's-', color=color, linewidth=1.5,
                   markersize=6, label=f"Naive (fixed $T$)")
        if not np.isnan(ot_mi):
            ax_mi.axhline(ot_mi, color='gray', linestyle='--', alpha=0.7,
                          label=f"OT (stability)")
        ax_mi.set_xlabel("Switch iteration $T$")
        ax_mi.set_ylabel("Mean iterations (successful)")
        ax_mi.set_xscale('log')
        ax_mi.set_xticks(tv)
        ax_mi.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax_mi.set_title(f"{label} — Mean Iterations vs $T$")
        ax_mi.legend()

    _savefig(fig, args.outdir, "ablation_naive")
    if args.show:
        plt.show()
    plt.close(fig)

    print("\nDone.")


if __name__ == "__main__":
    main()
