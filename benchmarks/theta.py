"""Theta convergence: cosine similarity of perturbation drift toward oracle.

Per-image workflow (SimBA only):
  1. Look up oracle target from winrate CSV (or run untargeted probe)
  2. Run targeted attack toward oracle -> get delta_oracle
  3. Run untargeted with reference_direction=delta_oracle -> cos_sim trajectory
  4. Run opportunistic with same reference -> trajectory + switch_iteration

Output: mean cosine similarity vs iteration (shaded +/- 1 std), two curves
(untargeted blue, opportunistic green), vertical dashed line at mean switch.

Usage:
    python analyze_theta.py --n-images 20 --show
    python analyze_theta.py --n-images 2 --show   # Smoke test
"""

import argparse
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from benchmarks.benchmark import load_benchmark_model, load_benchmark_image, get_true_label
from benchmarks.winrate import (
    select_images, lookup_oracle_targets, determine_oracle_target,
)
from src.attacks.simba import SimBA

# ===========================================================================
# Configuration
# ===========================================================================
MODEL_NAME = 'resnet50'
SOURCE = 'standard'
EPSILON = 8 / 255
STABILITY_THRESHOLD = 10
VAL_DIR = Path('data/imagenet/val')
RESULTS_DIR = Path('results')
WINRATE_CSV = RESULTS_DIR / 'benchmark_winrate.csv'
CACHE_DIR = RESULTS_DIR / 'theta_cache'
OUTDIR = RESULTS_DIR / 'figures_theta'


# ===========================================================================
# Style (matches other analysis scripts)
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
# Per-image cache
# ===========================================================================
def _cache_path(cache_dir, image_name, budget):
    return Path(cache_dir) / f"{Path(image_name).stem}_b{budget}.npz"


def _load_cached(cache_dir, image_name, budget):
    p = _cache_path(cache_dir, image_name, budget)
    if not p.exists():
        return None
    data = np.load(p, allow_pickle=True)
    return {
        'untargeted': (data['u_iters'], data['u_cos']),
        'opportunistic': (data['o_iters'], data['o_cos']),
        'switch_iteration': data['switch_iter'].item(),  # scalar or None
    }


def _save_cache(cache_dir, image_name, budget, result):
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    p = _cache_path(cache_dir, image_name, budget)
    u_iters, u_cos = result['untargeted']
    o_iters, o_cos = result['opportunistic']
    np.savez_compressed(
        p,
        u_iters=u_iters, u_cos=u_cos,
        o_iters=o_iters, o_cos=o_cos,
        switch_iter=np.array(result['switch_iteration'], dtype=object),
    )


# ===========================================================================
# Per-image pipeline
# ===========================================================================
def run_theta_pipeline(model, x, y_true, oracle_target, budget, device):
    """Run targeted + untargeted + opportunistic with cos_sim tracking.

    Returns:
        dict with keys 'untargeted' and 'opportunistic', each containing
        (iterations_array, cos_sim_array), plus 'switch_iteration' (int or None).
    """
    y_true_tensor = torch.tensor([y_true], device=device)
    target_tensor = torch.tensor([oracle_target], device=device)

    # Step 1: Targeted attack toward oracle -> get delta_oracle
    attack_t = SimBA(
        model=model, epsilon=EPSILON, max_iterations=budget,
        device=device, use_dct=True, pixel_range=(0.0, 1.0),
    )
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    x_adv_targeted = attack_t.generate(
        x, y_true_tensor,
        track_confidence=False,
        targeted=True,
        target_class=target_tensor,
        early_stop=True,
    )
    delta_oracle = (x_adv_targeted - x).squeeze(0)  # (C, H, W)

    # Step 2: Untargeted with reference_direction=delta_oracle
    attack_u = SimBA(
        model=model, epsilon=EPSILON, max_iterations=budget,
        device=device, use_dct=True, pixel_range=(0.0, 1.0),
    )
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    attack_u.generate(
        x, y_true_tensor,
        track_confidence=True,
        targeted=False,
        early_stop=True,
        reference_direction=delta_oracle,
    )
    hist_u = attack_u.confidence_history
    untargeted_data = (
        np.array(hist_u['cos_sim_iterations']),
        np.array(hist_u['cos_sim_to_ref']),
    )

    # Step 3: Opportunistic with same reference
    attack_o = SimBA(
        model=model, epsilon=EPSILON, max_iterations=budget,
        device=device, use_dct=True, pixel_range=(0.0, 1.0),
    )
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    attack_o.generate(
        x, y_true_tensor,
        track_confidence=True,
        targeted=False,
        early_stop=True,
        opportunistic=True,
        stability_threshold=STABILITY_THRESHOLD,
        reference_direction=delta_oracle,
    )
    hist_o = attack_o.confidence_history
    opportunistic_data = (
        np.array(hist_o['cos_sim_iterations']),
        np.array(hist_o['cos_sim_to_ref']),
    )
    switch_iter = hist_o.get('switch_iteration')

    return {
        'untargeted': untargeted_data,
        'opportunistic': opportunistic_data,
        'switch_iteration': switch_iter,
    }


# ===========================================================================
# Interpolation helper
# ===========================================================================
def interpolate_to_grid(all_trajectories, grid):
    """Interpolate trajectories to a common iteration grid.

    Args:
        all_trajectories: list of (iterations, cos_sim) tuples.
        grid: 1-D array of iteration values.

    Returns:
        (mean, std) arrays on the grid, computed only over images that
        have data at each grid point.
    """
    n = len(all_trajectories)
    matrix = np.full((n, len(grid)), np.nan)
    for i, (iters, cos) in enumerate(all_trajectories):
        if len(iters) == 0:
            continue
        matrix[i] = np.interp(grid, iters, cos, right=np.nan)

    with np.errstate(all='ignore'):
        mean = np.nanmean(matrix, axis=0)
        std = np.nanstd(matrix, axis=0)
    return mean, std


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Theta convergence analysis (cosine similarity of "
                    "perturbation drift toward oracle)"
    )
    parser.add_argument('--n-images', type=int, default=100,
                        help="Number of images (default: 100)")
    parser.add_argument('--budget', type=int, default=500,
                        help="Query budget per run (default: 500)")
    parser.add_argument('--image-seed', type=int, default=42,
                        help="Seed for image selection (default: 42)")
    parser.add_argument('--outdir', default=str(OUTDIR),
                        help="Output directory for figures")
    parser.add_argument('--show', action='store_true',
                        help="Show interactive plots")
    args = parser.parse_args()

    _setup_style()

    cache_dir = CACHE_DIR

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Device: {device}")
    print(f"Model: {MODEL_NAME} ({SOURCE})")
    print(f"Epsilon: {EPSILON:.6f} ({EPSILON * 255:.0f}/255)")
    print(f"Images: {args.n_images} (seed={args.image_seed})")
    print(f"Budget: {args.budget}")
    print()

    print(f"Selecting {args.n_images} images from {VAL_DIR}...")
    image_paths = select_images(VAL_DIR, args.n_images, args.image_seed)

    # Check which images need computation
    needs_compute = any(
        _load_cached(cache_dir, p.name, args.budget) is None
        for p in image_paths
    )

    model = None
    oracle_targets = None
    if needs_compute:
        print(f"Loading model: {MODEL_NAME} ({SOURCE})...")
        model = load_benchmark_model(MODEL_NAME, SOURCE, device)
        oracle_targets = lookup_oracle_targets(WINRATE_CSV)
    else:
        print("All images cached, skipping model load.")

    # Collect trajectories
    untargeted_trajs = []
    opportunistic_trajs = []
    switch_iters = []

    for idx, path in enumerate(image_paths):
        image_name = path.name

        # Try cache first
        cached = _load_cached(cache_dir, image_name, args.budget)
        if cached is not None:
            result = cached
            print(f"  [{idx+1}/{args.n_images}] {image_name}: cached")
        else:
            x = load_benchmark_image(path, device)
            y_true = get_true_label(model, x)
            y_true_tensor = torch.tensor([y_true], device=device)

            oracle = oracle_targets.get(('SimBA', image_name))
            if oracle is None:
                print(f"  [{idx+1}/{args.n_images}] {image_name}: "
                      f"running untargeted probe for oracle target...")
                oracle = determine_oracle_target(
                    model, 'SimBA', x, y_true_tensor, args.budget, device)

            print(f"  [{idx+1}/{args.n_images}] {image_name}: "
                  f"true={y_true}, oracle={oracle}")

            result = run_theta_pipeline(
                model, x, y_true, oracle, args.budget, device)
            _save_cache(cache_dir, image_name, args.budget, result)

        untargeted_trajs.append(result['untargeted'])

        # Before the switch, opportunistic = untargeted; stitch them together
        u_it, u_cs = result['untargeted']
        o_it, o_cs = result['opportunistic']
        sw = result['switch_iteration']
        if sw is not None and len(u_it) > 0 and len(o_it) > 0:
            pre_mask = u_it < sw
            post_mask = o_it >= sw
            stitched_iters = np.concatenate([u_it[pre_mask], o_it[post_mask]])
            stitched_cos = np.concatenate([u_cs[pre_mask], o_cs[post_mask]])
            opportunistic_trajs.append((stitched_iters, stitched_cos))
        else:
            opportunistic_trajs.append(result['opportunistic'])

        if sw is not None:
            switch_iters.append(sw)

        u_iters, u_cos = result['untargeted']
        o_iters, o_cos = result['opportunistic']
        u_final = u_cos[-1] if len(u_cos) > 0 else float('nan')
        o_final = o_cos[-1] if len(o_cos) > 0 else float('nan')
        sw = result['switch_iteration']
        sw_str = f", switch@{sw}" if sw is not None else ""
        print(f"    untargeted: {len(u_iters)} pts, final cos={u_final:.3f}")
        print(f"    opportunistic: {len(o_iters)} pts, "
              f"final cos={o_final:.3f}{sw_str}")

    # ---- Interpolate to common grid ----
    step = 10
    grid = np.arange(0, args.budget + 1, step)

    u_mean, u_std = interpolate_to_grid(untargeted_trajs, grid)
    o_mean, o_std = interpolate_to_grid(opportunistic_trajs, grid)

    mean_switch = np.mean(switch_iters) if switch_iters else None

    print(f"\nSwitch iterations collected: {len(switch_iters)}")
    if mean_switch is not None:
        print(f"Mean switch iteration: {mean_switch:.0f}")

    # ---- Figure ----
    os.makedirs(args.outdir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Untargeted (blue)
    color_u = "#4878CF"
    ax.plot(grid, u_mean, color=color_u, linewidth=1.5, label="Untargeted")
    ax.fill_between(grid, u_mean - u_std, u_mean + u_std,
                    color=color_u, alpha=0.15)

    # Opportunistic (green)
    color_o = "#6BA353"
    ax.plot(grid, o_mean, color=color_o, linewidth=1.5, label="Opportunistic")
    ax.fill_between(grid, o_mean - o_std, o_mean + o_std,
                    color=color_o, alpha=0.15)

    # Vertical line at mean switch iteration
    if mean_switch is not None:
        ax.axvline(mean_switch, color='gray', linestyle='--', alpha=0.7,
                   label=f"Mean switch ($i$={mean_switch:.0f})")

    # Trim x-axis to last iteration with valid data
    valid_mask = np.isfinite(u_mean) | np.isfinite(o_mean)
    x_max = grid[valid_mask][-1] if valid_mask.any() else args.budget

    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Cosine similarity to $\delta_{\mathrm{oracle}}$")
    ax.set_xlim(0, x_max)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_title(r"Perturbation Alignment with Oracle Direction (SimBA, ResNet-50)")

    _savefig(fig, args.outdir, "fig_theta")
    if args.show:
        plt.show()
    plt.close(fig)

    print("\nDone.")


if __name__ == '__main__':
    main()
