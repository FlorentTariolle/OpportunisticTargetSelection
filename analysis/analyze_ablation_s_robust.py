"""Analyze robust S-ablation results.

Reads benchmark_ablation_s_robust.csv and produces a 1x2 figure:
  - Left:  Success rate vs S (opportunistic line + baseline h-lines)
  - Right: Mean iterations vs S (opportunistic line + baseline h-lines)

Baselines (untargeted / targeted) are shown as horizontal dashed lines.

Usage:
    python analyze_ablation_s_robust.py                       # Default CSV + output
    python analyze_ablation_s_robust.py --show                # Interactive display
    python analyze_ablation_s_robust.py --csv results/custom.csv
"""

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ===========================================================================
# Style (matches analyze_winrate.py / analyze_ablation_s.py)
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
# Colors (matches analyze_winrate.py)
# ===========================================================================
MODE_COLORS = {
    "untargeted": "#4878CF",
    "targeted": "#E8873A",
    "opportunistic": "#6BA353",
}
MODE_LABELS = {
    "untargeted": "Untargeted",
    "targeted": "Targeted (oracle)",
    "opportunistic": "Opportunistic",
}


# ===========================================================================
# Statistics
# ===========================================================================
def _compute_baseline(df_mode):
    """Compute success rate and mean iterations for a baseline mode."""
    sr = df_mode["success"].mean() if len(df_mode) > 0 else 0.0
    succ = df_mode[df_mode["success"]]
    mi = succ["iterations"].mean() if len(succ) > 0 else np.nan
    return sr, mi


def _compute_opp_stats(df_opp, s_values):
    """Compute per-S success rate and mean iterations for opportunistic."""
    success_rates = []
    mean_iters = []
    for s in s_values:
        subset = df_opp[df_opp["s_value"] == s]
        sr = subset["success"].mean() if len(subset) > 0 else 0.0
        success_rates.append(sr)
        succ = subset[subset["success"]]
        mi = succ["iterations"].mean() if len(succ) > 0 else np.nan
        mean_iters.append(mi)
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
        description="Analyze robust S-ablation results"
    )
    parser.add_argument("--csv",
                        default="results/benchmark_ablation_s_robust.csv",
                        help="Path to robust ablation CSV")
    parser.add_argument("--outdir",
                        default="results/figures_ablation_s_robust",
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
    df["s_value"] = pd.to_numeric(df["s_value"], errors="coerce")
    df["switch_iteration"] = pd.to_numeric(
        df["switch_iteration"], errors="coerce"
    )

    os.makedirs(args.outdir, exist_ok=True)

    n_images = df["image"].nunique()
    print(f"Images: {n_images}")

    # ---- Baselines ----
    df_unt = df[df["mode"] == "untargeted"]
    df_tgt = df[df["mode"] == "targeted"]
    df_opp = df[df["mode"] == "opportunistic"]

    sr_unt, mi_unt = _compute_baseline(df_unt)
    sr_tgt, mi_tgt = _compute_baseline(df_tgt)

    print(f"\nUntargeted baseline:")
    n_succ = int(df_unt["success"].sum())
    mi_str = f"{mi_unt:.0f}" if not np.isnan(mi_unt) else "N/A"
    print(f"  {sr_unt:.1%} success ({n_succ}/{len(df_unt)}), "
          f"mean iters={mi_str}")

    print(f"\nTargeted (oracle) baseline:")
    n_succ = int(df_tgt["success"].sum())
    mi_str = f"{mi_tgt:.0f}" if not np.isnan(mi_tgt) else "N/A"
    print(f"  {sr_tgt:.1%} success ({n_succ}/{len(df_tgt)}), "
          f"mean iters={mi_str}")

    # ---- Opportunistic per S ----
    s_values = sorted(int(v) for v in df_opp["s_value"].dropna().unique())
    opp_sr, opp_mi = _compute_opp_stats(df_opp, s_values)
    best_s = _find_best_s(s_values, opp_sr, opp_mi)

    print(f"\nOpportunistic (S values: {s_values}):")
    for i, s in enumerate(s_values):
        subset = df_opp[df_opp["s_value"] == s]
        n = len(subset)
        n_succ = int(subset["success"].sum())
        avg_str = f"{opp_mi[i]:.0f}" if not np.isnan(opp_mi[i]) else "N/A"
        # Mean switch iteration for successful attacks
        succ_subset = subset[subset["success"]]
        sw = succ_subset["switch_iteration"].dropna()
        sw_str = f"{sw.mean():.0f}" if len(sw) > 0 else "N/A"
        print(f"  S={s:>2d}: {opp_sr[i]:.1%} success ({n_succ}/{n}), "
              f"mean iters={avg_str}, mean switch={sw_str}")
    print(f"  Optimal S: {best_s}")

    # ---- Figure: 1x2 ----
    fig, (ax_sr, ax_mi) = plt.subplots(1, 2, figsize=(10, 4.5))

    c_opp = MODE_COLORS["opportunistic"]
    c_unt = MODE_COLORS["untargeted"]
    c_tgt = MODE_COLORS["targeted"]

    # --- Left: Success rate vs S ---
    ax_sr.plot(s_values, opp_sr, 'o-', color=c_opp, linewidth=1.5,
               markersize=6, label=MODE_LABELS["opportunistic"], zorder=3)
    ax_sr.axhline(sr_unt, color=c_unt, linestyle='--', linewidth=1.2,
                  label=f'{MODE_LABELS["untargeted"]} ({sr_unt:.1%})')
    ax_sr.axhline(sr_tgt, color=c_tgt, linestyle='--', linewidth=1.2,
                  label=f'{MODE_LABELS["targeted"]} ({sr_tgt:.1%})')
    ax_sr.axvline(best_s, color='gray', linestyle=':', alpha=0.6,
                  label=f'Best $S={best_s}$')
    ax_sr.set_xlabel("Stability threshold $S$")
    ax_sr.set_ylabel("Success rate")
    ax_sr.set_xticks(s_values)
    ax_sr.set_ylim(-0.02, 1.02)
    ax_sr.legend(loc="lower left", framealpha=0.9)
    ax_sr.set_title("Success Rate vs $S$")

    # --- Right: Mean iterations vs S ---
    ax_mi.plot(s_values, opp_mi, 's-', color=c_opp, linewidth=1.5,
               markersize=6, label=MODE_LABELS["opportunistic"], zorder=3)
    if not np.isnan(mi_unt):
        ax_mi.axhline(mi_unt, color=c_unt, linestyle='--', linewidth=1.2,
                      label=f'{MODE_LABELS["untargeted"]} ({mi_unt:.0f})')
    if not np.isnan(mi_tgt):
        ax_mi.axhline(mi_tgt, color=c_tgt, linestyle='--', linewidth=1.2,
                      label=f'{MODE_LABELS["targeted"]} ({mi_tgt:.0f})')
    ax_mi.axvline(best_s, color='gray', linestyle=':', alpha=0.6)
    ax_mi.set_xlabel("Stability threshold $S$")
    ax_mi.set_ylabel("Mean iterations (successful)")
    ax_mi.set_xticks(s_values)
    ax_mi.legend(loc="upper left", framealpha=0.9)
    ax_mi.set_title("Mean Iterations vs $S$")

    _savefig(fig, args.outdir, "fig_ablation_s_robust")
    if args.show:
        plt.show()
    plt.close(fig)

    print("\nDone.")


if __name__ == "__main__":
    main()
