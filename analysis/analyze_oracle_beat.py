"""
OT-beats-oracle rate analysis

Compares opportunistic-target (OT) vs oracle-target iterations for cases
where both succeed. Reports how often OT converges faster, ties, or is slower.

Usage:
    python analyze_oracle_beat.py                      # Standard models (default)
    python analyze_oracle_beat.py --source robust      # Robust models
    python analyze_oracle_beat.py --show               # Interactive display
"""

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ===========================================================================
# Style configuration (mirrors analyze_benchmark.py)
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


MODE_ORDER = ["untargeted", "targeted", "opportunistic"]

MODEL_ORDERS = {
    "standard": ["resnet18", "resnet50", "vgg16", "alexnet"],
    "robust": ["Salman2020Do_R18", "Salman2020Do_R50"],
}

OUTCOME_COLORS = {
    "OT wins": "#6BA353",   # green
    "Tie": "#AAAAAA",       # grey
    "Oracle wins": "#E8873A",  # orange
}


# ===========================================================================
# Data loading
# ===========================================================================
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["iterations"] = pd.to_numeric(df["iterations"])
    df["success"] = df["success"].map(
        {"True": True, "False": False, True: True, False: False}
    )
    df["mode"] = pd.Categorical(df["mode"], categories=MODE_ORDER, ordered=True)
    return df


def _savefig(fig, outdir: str, name: str):
    fig.savefig(os.path.join(outdir, f"{name}.png"), bbox_inches="tight")
    fig.savefig(os.path.join(outdir, f"{name}.pdf"), bbox_inches="tight")
    print(f"  Saved {name}.png / .pdf")


# ===========================================================================
# Core analysis
# ===========================================================================
def compute_oracle_beat(df: pd.DataFrame) -> pd.DataFrame:
    """Pair opportunistic vs targeted rows and classify outcomes."""
    ot = df[(df["mode"] == "opportunistic") & (df["success"] == True)].copy()
    oracle = df[(df["mode"] == "targeted") & (df["success"] == True)].copy()

    key = ["model", "method", "image", "epsilon", "seed"]
    merged = ot.merge(oracle, on=key, suffixes=("_ot", "_oracle"))

    merged["outcome"] = np.where(
        merged["iterations_ot"] < merged["iterations_oracle"], "OT wins",
        np.where(merged["iterations_ot"] == merged["iterations_oracle"],
                 "Tie", "Oracle wins")
    )
    merged["iter_diff"] = merged["iterations_oracle"] - merged["iterations_ot"]
    return merged


def summarize(merged: pd.DataFrame) -> pd.DataFrame:
    """Compute outcome fractions by model and method."""
    rows = []
    for (model, method), grp in merged.groupby(["model", "method"]):
        n = len(grp)
        for outcome in ["OT wins", "Tie", "Oracle wins"]:
            count = (grp["outcome"] == outcome).sum()
            rows.append({
                "model": model, "method": method,
                "outcome": outcome, "count": count, "n": n,
                "fraction": count / n if n > 0 else 0,
            })

    # Overall row
    n_total = len(merged)
    for outcome in ["OT wins", "Tie", "Oracle wins"]:
        count = (merged["outcome"] == outcome).sum()
        rows.append({
            "model": "Overall", "method": "All",
            "outcome": outcome, "count": count, "n": n_total,
            "fraction": count / n_total if n_total > 0 else 0,
        })

    return pd.DataFrame(rows)


# ===========================================================================
# Figure
# ===========================================================================
def fig_oracle_beat(summary: pd.DataFrame, outdir: str, model_order: list[str]):
    """Stacked horizontal bar chart of OT-vs-oracle outcomes."""
    groups = summary[summary["model"] != "Overall"].copy()
    groups["label"] = groups["model"] + " / " + groups["method"]

    # Order by model_order then method
    order_map = {m: i for i, m in enumerate(model_order)}
    groups["sort_key"] = groups["model"].map(order_map).fillna(99)
    labels_order = (
        groups.drop_duplicates("label")
        .sort_values(["sort_key", "method"])["label"]
        .tolist()
    )

    fig, ax = plt.subplots(figsize=(8, max(3, len(labels_order) * 0.5 + 1)))

    y_pos = np.arange(len(labels_order))
    lefts = np.zeros(len(labels_order))

    for outcome in ["OT wins", "Tie", "Oracle wins"]:
        widths = []
        for label in labels_order:
            row = groups[(groups["label"] == label) & (groups["outcome"] == outcome)]
            widths.append(row["fraction"].values[0] if len(row) else 0)
        widths = np.array(widths)
        ax.barh(y_pos, widths, left=lefts, height=0.6,
                color=OUTCOME_COLORS[outcome], label=outcome)
        # Annotate percentages
        for i, (w, l) in enumerate(zip(widths, lefts)):
            if w > 0.05:
                ax.text(l + w / 2, y_pos[i], f"{w:.0%}",
                        ha="center", va="center", fontsize=9, fontweight="bold")
        lefts += widths

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_order)
    ax.set_xlabel("Fraction of paired successes")
    ax.set_title("OT vs Oracle: iteration comparison (both succeed)")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.invert_yaxis()

    _savefig(fig, outdir, "oracle_beat_rate")
    plt.close(fig)


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Analyze how often OT beats the oracle target in iterations."
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
    args = parser.parse_args()

    if args.csv is None:
        args.csv = f"results/benchmark_{args.source}.csv"
    if args.outdir is None:
        args.outdir = f"results/figures/{args.source}"

    if not args.show:
        matplotlib.use("Agg")

    _setup_style()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Loading data from {args.csv} ...")
    df = load_data(args.csv)
    print(f"  {len(df)} rows")

    model_order = MODEL_ORDERS[args.source]

    print("\n=== OT-beats-Oracle Analysis ===")
    merged = compute_oracle_beat(df)
    print(f"  {len(merged)} paired successes (both OT and oracle succeed)")

    if merged.empty:
        print("  No paired successes found. Nothing to analyze.")
        return

    summary = summarize(merged)

    # Print summary table
    print("\n--- Breakdown by model/method ---")
    pivot = summary.pivot_table(
        index=["model", "method"], columns="outcome",
        values=["count", "fraction"], aggfunc="first"
    )
    pivot = pivot.reindex(columns=["OT wins", "Tie", "Oracle wins"], level=1)
    print(pivot.to_string())

    # Overall
    overall = summary[summary["model"] == "Overall"]
    print("\n--- Overall ---")
    for _, row in overall.iterrows():
        print(f"  {row['outcome']}: {row['count']}/{row['n']} ({row['fraction']:.1%})")

    # Median iteration savings when OT wins
    ot_wins = merged[merged["outcome"] == "OT wins"]
    if not ot_wins.empty:
        med_saving = ot_wins["iter_diff"].median()
        mean_saving = ot_wins["iter_diff"].mean()
        print(f"\n  When OT wins: median saving = {med_saving:.0f} iters, "
              f"mean saving = {mean_saving:.1f} iters")

    # Save CSV
    csv_out = os.path.join("results", "oracle_beat_summary.csv")
    summary.to_csv(csv_out, index=False)
    print(f"\n  Summary saved to {csv_out}")

    # Generate figure
    fig_oracle_beat(summary, args.outdir, model_order)

    if args.show:
        plt.show()

    print("\nDone.")


if __name__ == "__main__":
    main()
