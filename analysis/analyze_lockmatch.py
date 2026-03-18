"""
Lock-Match Correlation Analysis (Issue #13)

Analyzes whether OT's success depends on finding the oracle class or any
viable class.  Reads both benchmark_standard.csv and benchmark_winrate.csv,
computes point-biserial correlations and contingency tables, and generates
publication figures.

Usage:
    python analyze_lockmatch.py                  # Default (no display)
    python analyze_lockmatch.py --show           # Interactive display
    python analyze_lockmatch.py --outdir results/figures_lockmatch
"""

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr


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


METHOD_COLORS = {"SimBA": "#4878CF", "SquareAttack": "#D65F5F", "Bandits": "#6BA353"}


# ===========================================================================
# Data loading
# ===========================================================================
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["iterations"] = pd.to_numeric(df["iterations"])
    df["success"] = df["success"].map(
        {"True": True, "False": False, True: True, False: False}
    )
    df["locked_class"] = pd.to_numeric(df["locked_class"], errors="coerce")
    df["adversarial_class"] = pd.to_numeric(df["adversarial_class"], errors="coerce")
    return df


def build_pairs(df: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    """Pair opportunistic (with lock data) and untargeted runs."""
    opp = df[(df["mode"] == "opportunistic") & df["locked_class"].notna()][
        key_cols + ["iterations", "success", "locked_class", "adversarial_class"]
    ].copy()
    opp.columns = [c + "_opp" if c not in key_cols else c for c in opp.columns]

    unt = df[df["mode"] == "untargeted"][
        key_cols + ["iterations", "success", "adversarial_class"]
    ].copy()
    unt.columns = [c + "_unt" if c not in key_cols else c for c in unt.columns]

    merged = opp.merge(unt, on=key_cols)
    merged["lock_match"] = merged["locked_class_opp"] == merged["adversarial_class_unt"]
    merged["savings"] = (
        (merged["iterations_unt"] - merged["iterations_opp"]) / merged["iterations_unt"]
    )
    return merged


# ===========================================================================
# Helpers
# ===========================================================================
def _savefig(fig, outdir: str, name: str):
    fig.savefig(os.path.join(outdir, f"{name}.png"), bbox_inches="tight")
    fig.savefig(os.path.join(outdir, f"{name}.pdf"), bbox_inches="tight")
    print(f"  Saved {name}.png / .pdf")


def _sig_str(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# ===========================================================================
# Console output
# ===========================================================================
def print_summary(pairs: pd.DataFrame, label: str):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    n = len(pairs)
    n_match = pairs["lock_match"].sum()
    print(f"  Paired runs: {n}  (match={n_match}, mismatch={n - n_match})")
    print(f"  Lock-match rate: {pairs['lock_match'].mean():.1%}")

    # Contingency: lock_match × OT success
    print(f"\n  Contingency (lock_match × OT success):")
    ct = pd.crosstab(
        pairs["lock_match"].map({True: "Match", False: "Mismatch"}),
        pairs["success_opp"].map({True: "OT success", False: "OT fail"}),
    )
    print(ct.to_string().replace("\n", "\n  "))

    print(f"\n  OT success | Match: {pairs[pairs['lock_match']]['success_opp'].mean():.1%}"
          f"  | Mismatch: {pairs[~pairs['lock_match']]['success_opp'].mean():.1%}")

    # Both-succeed correlation
    both = pairs[pairs["success_opp"] & pairs["success_unt"]]
    print(f"\n  Both-succeed subset (N={len(both)}):")
    if len(both) > 5:
        r, p = pointbiserialr(both["lock_match"].astype(int), both["savings"])
        print(f"    Point-biserial: r={r:.3f}, p={p:.2e} {_sig_str(p)}")
        print(f"    Match savings:    {both[both['lock_match']]['savings'].mean():+.1%}")
        print(f"    Mismatch savings: {both[~both['lock_match']]['savings'].mean():+.1%}")

    for method in sorted(both["method"].unique()):
        sub = both[both["method"] == method]
        if len(sub) > 5:
            r, p = pointbiserialr(sub["lock_match"].astype(int), sub["savings"])
            print(f"    {method:14s} (N={len(sub):3d}): r={r:.3f}, p={p:.2e} {_sig_str(p)}"
                  f"  match={sub[sub['lock_match']]['savings'].mean():+.1%}"
                  f"  mismatch={sub[~sub['lock_match']]['savings'].mean():+.1%}")


# ===========================================================================
# Figures
# ===========================================================================
def fig_lockmatch_savings(std_pairs: pd.DataFrame, win_pairs: pd.DataFrame,
                          outdir: str):
    """Grouped bar: mean savings by lock-match, per method, per benchmark."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    for ax, (pairs, title, budget) in zip(axes, [
        (std_pairs, "50-image benchmark (10K budget)", "10K"),
        (win_pairs, "100-image benchmark (15K budget)", "15K"),
    ]):
        both = pairs[pairs["success_opp"] & pairs["success_unt"]]
        methods = sorted(both["method"].unique())
        x = np.arange(len(methods))
        width = 0.3

        for j, match_val in enumerate([True, False]):
            label = "Match" if match_val else "Mismatch"
            vals, errs = [], []
            for method in methods:
                sub = both[(both["method"] == method) & (both["lock_match"] == match_val)]
                vals.append(sub["savings"].mean() * 100 if len(sub) > 0 else 0)
                errs.append(sub["savings"].sem() * 100 if len(sub) > 1 else 0)
            color = "#6BA353" if match_val else "#D64541"
            bars = ax.bar(x + (j - 0.5) * width, vals, width, yerr=errs,
                          color=color, edgecolor="white", linewidth=0.5,
                          label=label, capsize=3)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (2 if v >= 0 else -4),
                        f"{v:+.1f}%", ha="center", fontsize=9)

        ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_title(title)
        ax.legend(loc="lower left")

    axes[0].set_ylabel("Mean Query Savings (%)")
    fig.suptitle("Query Savings by Lock-Match (both-succeed subset)", fontsize=13)
    _savefig(fig, outdir, "fig_lockmatch_savings")
    return fig


def fig_lockmatch_success(std_pairs: pd.DataFrame, win_pairs: pd.DataFrame,
                          outdir: str):
    """Bar chart: OT success rate split by lock-match, per benchmark."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    for ax, (pairs, title) in zip(axes, [
        (std_pairs, "50-image (10K)"),
        (win_pairs, "100-image (15K)"),
    ]):
        methods = sorted(both["method"].unique())
        x = np.arange(len(methods))
        width = 0.3
        for j, match_val in enumerate([True, False]):
            label = "Match" if match_val else "Mismatch"
            vals, ns = [], []
            for method in methods:
                sub = pairs[(pairs["method"] == method) & (pairs["lock_match"] == match_val)]
                vals.append(sub["success_opp"].mean() * 100 if len(sub) > 0 else 0)
                ns.append(len(sub))
            color = "#6BA353" if match_val else "#D64541"
            bars = ax.bar(x + (j - 0.5) * width, vals, width,
                          color=color, edgecolor="white", linewidth=0.5,
                          label=label)
            for bar, v, n in zip(bars, vals, ns):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 1.5,
                        f"{v:.0f}%\n(N={n})", ha="center", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_title(title)
        ax.set_ylim(0, 115)
        ax.legend()

    axes[0].set_ylabel("OT Success Rate (%)")
    fig.suptitle("OT Success Rate by Lock-Match", fontsize=13)
    _savefig(fig, outdir, "fig_lockmatch_success")
    return fig


def fig_lockmatch_correlation(std_pairs: pd.DataFrame, win_pairs: pd.DataFrame,
                              outdir: str):
    """Strip plot: per-image savings by lock-match, with correlation annotated."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    for ax, (pairs, title) in zip(axes, [
        (std_pairs, "50-image (10K)"),
        (win_pairs, "100-image (15K)"),
    ]):
        both = pairs[pairs["success_opp"] & pairs["success_unt"]]
        if len(both) < 5:
            ax.set_title(f"{title}\n(insufficient data)")
            continue

        for method, marker in [("SimBA", "o"), ("SquareAttack", "s")]:
            sub = both[both["method"] == method]
            jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(sub))
            x_vals = sub["lock_match"].astype(int) + jitter
            ax.scatter(x_vals, sub["savings"] * 100, alpha=0.5, s=25,
                       marker=marker, color=METHOD_COLORS[method],
                       label=method, edgecolors="none")

        r, p = pointbiserialr(both["lock_match"].astype(int), both["savings"])
        ax.set_title(f"{title}\nr = {r:.3f}, p = {p:.2e} {_sig_str(p)}")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Mismatch", "Match"])
        ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
        ax.legend(loc="upper left", fontsize=8)

    axes[0].set_ylabel("Query Savings (%)")
    fig.suptitle("Per-Image Savings by Lock-Match", fontsize=13)
    _savefig(fig, outdir, "fig_lockmatch_correlation")
    return fig


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Lock-match correlation analysis")
    parser.add_argument("--std-csv", default="results/benchmark_standard.csv")
    parser.add_argument("--win-csv", default="results/benchmark_winrate.csv")
    parser.add_argument("--outdir", default="results/figures_lockmatch")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if not args.show:
        matplotlib.use("Agg")

    _setup_style()
    os.makedirs(args.outdir, exist_ok=True)

    # --- Load & pair ---
    std_key = ["model", "method", "epsilon", "seed", "image"]
    win_key = ["method", "image"]  # winrate has no model/epsilon/seed cols

    std_df = load_csv(args.std_csv)
    std_pairs = build_pairs(std_df, std_key)

    win_df = load_csv(args.win_csv)
    win_pairs = build_pairs(win_df, win_key)

    # --- Console summary ---
    print_summary(std_pairs, "Standard Benchmark (50 images, 10K budget)")
    print_summary(win_pairs, "Winrate Benchmark (100 images, 15K budget)")

    # --- Figures ---
    print("\nGenerating figures...")
    fig_lockmatch_savings(std_pairs, win_pairs, args.outdir)
    fig_lockmatch_success(std_pairs, win_pairs, args.outdir)
    fig_lockmatch_correlation(std_pairs, win_pairs, args.outdir)

    if args.show:
        plt.show()

    print("\nDone.")


if __name__ == "__main__":
    main()
