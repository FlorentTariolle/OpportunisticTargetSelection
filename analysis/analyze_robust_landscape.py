"""Robust landscape analysis (Issue #18).

Loads per-iteration confidence_history JSON files from benchmark_landscape.py
and compares standard vs robust model landscapes via:
  1. Top-10 entropy over time (how diffuse is the class distribution?)
  2. Class volatility (how often does the top-1 non-true class change?)
  3. Confidence gap at lock-in (how strong is the lock-in signal?)

Usage:
    python analyze_robust_landscape.py                    # Default
    python analyze_robust_landscape.py --show             # Interactive
    python analyze_robust_landscape.py --data-dir results/landscape
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu


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


SOURCE_COLORS = {
    "standard": "#4878CF",
    "robust": "#D65F5F",
}
SOURCE_LABELS = {
    "standard": "Standard (ResNet-50)",
    "robust": "Robust (Salman2020Do\\_R50)",
}

# Minimum number of sampled iterations to include a run in trajectory analysis
MIN_SAMPLES = 10


# ===========================================================================
# Data loading
# ===========================================================================
def load_json_runs(data_dir: str) -> list[dict]:
    """Load all JSON files from the landscape data directory."""
    runs = []
    for path in sorted(Path(data_dir).glob("*.json")):
        with open(path) as f:
            runs.append(json.load(f))
    return runs


def filter_runs(runs: list[dict], source: str = None,
                mode: str = None) -> list[dict]:
    """Filter runs by source and/or mode."""
    out = runs
    if source:
        out = [r for r in out if r["source"] == source]
    if mode:
        out = [r for r in out if r["mode"] == mode]
    return out


# ===========================================================================
# Entropy computation
# ===========================================================================
def compute_entropy_trajectory(run: dict) -> tuple[list, list]:
    """Compute Shannon entropy of top-10 class distribution per iteration.

    Returns (iterations, entropies) lists.
    """
    ch = run["confidence_history"]
    iterations = ch["iterations"]
    top_classes = ch.get("top_classes", [])

    if not top_classes:
        return [], []

    entropies = []
    # top_classes may be shorter than iterations (only logged before lock-in)
    iters_out = iterations[:len(top_classes)]
    for tc in top_classes:
        probs = np.array(list(tc.values()))
        probs = probs[probs > 0]
        if len(probs) == 0:
            entropies.append(0.0)
        else:
            # Normalize to distribution over top-10
            probs = probs / probs.sum()
            entropies.append(-np.sum(probs * np.log2(probs)))
    return iters_out, entropies


# ===========================================================================
# Volatility computation
# ===========================================================================
def compute_volatility(run: dict) -> float:
    """Fraction of consecutive samples where top-1 class changes."""
    ids = run["confidence_history"]["max_other_class_id"]
    if len(ids) < 2:
        return float("nan")
    changes = sum(1 for a, b in zip(ids[:-1], ids[1:]) if a != b)
    return changes / (len(ids) - 1)


# ===========================================================================
# Confidence gap at lock-in
# ===========================================================================
def compute_lockin_gap(run: dict) -> float | None:
    """Confidence gap between locked class and runner-up at lock-in.

    Returns None if no lock-in occurred or insufficient data.
    """
    switch_iter = run.get("switch_iteration")
    if switch_iter is None:
        return None

    ch = run["confidence_history"]
    top_classes = ch.get("top_classes", [])
    iterations = ch["iterations"]

    if not top_classes:
        return None

    # Find the top_classes entry closest to switch_iteration
    tc_iters = iterations[:len(top_classes)]
    best_idx = None
    best_dist = float("inf")
    for i, it in enumerate(tc_iters):
        dist = abs(it - switch_iter)
        if dist < best_dist:
            best_dist = dist
            best_idx = i

    if best_idx is None:
        return None

    tc = top_classes[best_idx]
    confs = sorted(tc.values(), reverse=True)
    if len(confs) < 2:
        return None
    return confs[0] - confs[1]


# ===========================================================================
# Console summary
# ===========================================================================
def print_summary(runs: list[dict]):
    print(f"\n{'='*70}")
    print("  Landscape Analysis Summary")
    print(f"{'='*70}")

    for source in ["standard", "robust"]:
        opp = filter_runs(runs, source=source, mode="opportunistic")
        unt = filter_runs(runs, source=source, mode="untargeted")

        # Volatility (all modes)
        all_runs = opp + unt
        vols = [compute_volatility(r) for r in all_runs
                if not np.isnan(compute_volatility(r))]

        # Entropy (opportunistic only, need top_classes)
        entropies_mean = []
        for r in opp:
            _, ents = compute_entropy_trajectory(r)
            if ents:
                entropies_mean.append(np.mean(ents))

        # Lock-in gap (opportunistic only)
        gaps = [g for r in opp if (g := compute_lockin_gap(r)) is not None]

        # Lock-in rate
        n_locked = sum(1 for r in opp if r.get("switch_iteration") is not None)

        print(f"\n  {SOURCE_LABELS[source]}:")
        print(f"    Runs: {len(opp)} opportunistic, {len(unt)} untargeted")
        print(f"    Lock-in rate: {n_locked}/{len(opp)} "
              f"({100*n_locked/max(len(opp),1):.0f}%)")
        if vols:
            print(f"    Volatility: {np.mean(vols):.3f} +/- {np.std(vols):.3f} "
                  f"(N={len(vols)})")
        if entropies_mean:
            print(f"    Mean entropy: {np.mean(entropies_mean):.3f} +/- "
                  f"{np.std(entropies_mean):.3f} (N={len(entropies_mean)})")
        if gaps:
            print(f"    Lock-in gap: {np.mean(gaps):.4f} +/- {np.std(gaps):.4f} "
                  f"(N={len(gaps)})")


def print_tests(runs: list[dict]):
    """Mann-Whitney U tests: standard vs robust."""
    print(f"\n  Statistical tests (Mann-Whitney U):")

    for metric_name, compute_fn, mode_filter in [
        ("Volatility", compute_volatility, None),
        ("Mean entropy", lambda r: np.mean(compute_entropy_trajectory(r)[1])
         if compute_entropy_trajectory(r)[1] else np.nan, "opportunistic"),
        ("Lock-in gap", lambda r: compute_lockin_gap(r) or np.nan,
         "opportunistic"),
    ]:
        std_runs = filter_runs(runs, source="standard",
                               mode=mode_filter)
        rob_runs = filter_runs(runs, source="robust",
                               mode=mode_filter)

        std_vals = [v for r in std_runs
                    if not np.isnan(v := compute_fn(r))]
        rob_vals = [v for r in rob_runs
                    if not np.isnan(v := compute_fn(r))]

        if len(std_vals) < 3 or len(rob_vals) < 3:
            print(f"    {metric_name}: insufficient data "
                  f"(std={len(std_vals)}, rob={len(rob_vals)})")
            continue

        stat, p = mannwhitneyu(std_vals, rob_vals, alternative="two-sided")
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"    {metric_name}: U={stat:.0f}, p={p:.2e} {sig} "
              f"(std: {np.mean(std_vals):.4f}, rob: {np.mean(rob_vals):.4f})")


# ===========================================================================
# Figures
# ===========================================================================
def fig_entropy_trajectory(runs: list[dict], outdir: str):
    """Entropy over time: standard vs robust (mean +/- std)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for source in ["standard", "robust"]:
        opp = filter_runs(runs, source=source, mode="opportunistic")
        # Collect trajectories, interpolate to common iteration axis
        trajectories = []
        for r in opp:
            iters, ents = compute_entropy_trajectory(r)
            if len(iters) >= MIN_SAMPLES:
                trajectories.append((np.array(iters), np.array(ents)))

        if not trajectories:
            continue

        # Create common x-axis (union of all iteration points up to max)
        max_iter = max(t[0][-1] for t in trajectories)
        x_common = np.arange(0, min(max_iter + 1, 1001))

        # Interpolate each trajectory onto common axis
        interp_ents = []
        for iters, ents in trajectories:
            interp = np.interp(x_common, iters, ents)
            interp_ents.append(interp)

        interp_ents = np.array(interp_ents)
        mean = interp_ents.mean(axis=0)
        std = interp_ents.std(axis=0)

        color = SOURCE_COLORS[source]
        ax.plot(x_common, mean, color=color, linewidth=1.5,
                label=f"{SOURCE_LABELS[source]} (N={len(trajectories)})")
        ax.fill_between(x_common, mean - std, mean + std,
                        color=color, alpha=0.15)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Top-10 entropy (bits)")
    ax.set_title("Class Distribution Entropy Over Time")
    ax.legend(loc="upper right", framealpha=0.9)
    _savefig(fig, outdir, "fig_landscape_entropy")
    return fig


def fig_volatility(runs: list[dict], outdir: str):
    """Volatility comparison: standard vs robust (box plot)."""
    fig, ax = plt.subplots(figsize=(6, 4.5))

    data = []
    labels = []
    colors = []
    for source in ["standard", "robust"]:
        all_runs = filter_runs(runs, source=source)
        vols = [compute_volatility(r) for r in all_runs
                if not np.isnan(compute_volatility(r))
                and len(r["confidence_history"]["max_other_class_id"]) >= MIN_SAMPLES]
        if vols:
            data.append(vols)
            labels.append(SOURCE_LABELS[source])
            colors.append(SOURCE_COLORS[source])

    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)

    ax.set_ylabel("Volatility (fraction of rank changes)")
    ax.set_title("Top-1 Class Ranking Volatility")
    _savefig(fig, outdir, "fig_landscape_volatility")
    return fig


def fig_lockin_gap(runs: list[dict], outdir: str):
    """Lock-in confidence gap: standard vs robust (box plot)."""
    fig, ax = plt.subplots(figsize=(6, 4.5))

    data = []
    labels = []
    colors = []
    for source in ["standard", "robust"]:
        opp = filter_runs(runs, source=source, mode="opportunistic")
        gaps = [g for r in opp if (g := compute_lockin_gap(r)) is not None]
        if gaps:
            data.append(gaps)
            labels.append(SOURCE_LABELS[source])
            colors.append(SOURCE_COLORS[source])

    if not data:
        print("  No lock-in gap data available, skipping figure")
        plt.close(fig)
        return None

    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)

    ax.set_ylabel("Confidence gap (locked vs runner-up)")
    ax.set_title("Confidence Gap at Lock-In")
    _savefig(fig, outdir, "fig_landscape_gap")
    return fig


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Robust landscape analysis"
    )
    parser.add_argument("--data-dir", default="results/landscape")
    parser.add_argument("--outdir", default="results/figures_landscape")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if not args.show:
        matplotlib.use("Agg")

    _setup_style()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Loading data from {args.data_dir}")
    runs = load_json_runs(args.data_dir)
    print(f"  {len(runs)} runs loaded")

    if not runs:
        print("No data found. Run benchmark_landscape.py first.")
        return

    print_summary(runs)
    print_tests(runs)

    print("\nGenerating figures...")
    fig_entropy_trajectory(runs, args.outdir)
    fig_volatility(runs, args.outdir)
    fig_lockin_gap(runs, args.outdir)

    if args.show:
        plt.show()

    print("\nDone.")


if __name__ == "__main__":
    main()
