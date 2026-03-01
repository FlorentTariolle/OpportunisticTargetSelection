"""Margin-loss ablation benchmark (Issue #12).

Runs SquareAttack with margin loss on ResNet-50 (standard) across 100 random
ImageNet val images in two modes: untargeted and opportunistic (S=8).
Comparison with CE-loss modes (CE+OT and CE oracle) comes from
benchmark_winrate.csv, loaded at analysis time.

Split by images for parallel execution:
  --part 1  runs first half of images
  --part 2  runs second half of images

Usage:
    python benchmark_margin.py --part 1
    python benchmark_margin.py --part 2
    python benchmark_margin.py --part 1 --n-images 2   # smoke test
    python benchmark_margin.py --clear --part 1
"""

import argparse
import csv
import random
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmark import load_benchmark_model, load_benchmark_image, get_true_label
from src.attacks.square import SquareAttack

# ===========================================================================
# Configuration
# ===========================================================================
MODEL_NAME = 'resnet50'
SOURCE = 'standard'
EPSILON = 8 / 255
MAX_BUDGET = 15_000
STABILITY_THRESHOLD = 8  # Validated S* for SquareAttack standard
VAL_DIR = Path('data/imagenet/val')
RESULTS_DIR = Path('results')
CSV_PATH = RESULTS_DIR / 'benchmark_margin.csv'

CSV_COLUMNS = [
    'method', 'image', 'true_label', 'mode', 'budget', 'loss',
    'iterations', 'success', 'adversarial_class',
    'switch_iteration', 'locked_class', 'timestamp',
]


# ===========================================================================
# Image selection (mirrors benchmark_winrate.py for identical image set)
# ===========================================================================
def select_images(val_dir: Path, n: int, seed: int) -> list[Path]:
    """Select n random images from ImageNet val directory."""
    all_images = sorted(
        list(val_dir.glob('**/*.JPEG')) + list(val_dir.glob('**/*.jpeg'))
    )
    seen = set()
    unique = []
    for p in all_images:
        key = str(p).lower()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    all_images = unique

    if len(all_images) < n:
        raise ValueError(
            f"Found only {len(all_images)} images in {val_dir}, need {n}. "
            f"Make sure data/imagenet/val/ has ImageFolder structure."
        )
    rng = random.Random(seed)
    selected = rng.sample(all_images, n)
    return sorted(selected)


# ===========================================================================
# CSV I/O
# ===========================================================================
def append_row(row: dict, path: Path):
    """Append a single row to the CSV file."""
    file_exists = path.exists() and path.stat().st_size > 0
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def load_existing_keys(path: Path) -> set:
    """Load existing (image, mode) keys for resume."""
    keys = set()
    if not path.exists():
        return keys
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            keys.add((row['image'], row['mode']))
    return keys


# ===========================================================================
# Attack helper
# ===========================================================================
def run_attack(model, x, y_true_tensor, mode, budget, device):
    """Run SquareAttack with margin loss in the given mode.

    Args:
        model: The model (accepts [0,1] input).
        x: Input tensor (1, 3, 224, 224) in [0,1].
        y_true_tensor: True label tensor (1,).
        mode: 'untargeted' or 'opportunistic'.
        budget: Max iterations.
        device: torch device.

    Returns:
        dict with iterations, success, adversarial_class,
        switch_iteration, locked_class.
    """
    is_opportunistic = (mode == 'opportunistic')
    y_true_int = y_true_tensor.item()

    attack = SquareAttack(
        model=model, epsilon=EPSILON, max_iterations=budget,
        device=device, loss='margin', normalize=False, seed=0,
    )

    x_adv = attack.generate(
        x, y_true_tensor,
        track_confidence=True,
        targeted=False,
        early_stop=True,
        opportunistic=is_opportunistic,
        stability_threshold=STABILITY_THRESHOLD,
    )

    # Extract iteration count
    conf_hist = attack.confidence_history
    if conf_hist and conf_hist.get('iterations'):
        iterations = conf_hist['iterations'][-1]
    else:
        iterations = budget

    # Check success + final prediction
    with torch.no_grad():
        logits = model(x_adv)
        pred = logits.argmax(dim=1).item()
    success = (pred != y_true_int)

    # Switch info (opportunistic only)
    switch_iter = None
    locked_cls = None
    if conf_hist:
        switch_iter = conf_hist.get('switch_iteration')
        locked_cls = conf_hist.get('locked_class')

    return {
        'iterations': iterations,
        'success': success,
        'adversarial_class': pred,
        'switch_iteration': switch_iter,
        'locked_class': locked_cls,
    }


def make_row(image_name, true_label, mode, budget, result):
    """Build a CSV row dict from attack result."""
    return {
        'method': 'SquareAttack',
        'image': image_name,
        'true_label': true_label,
        'mode': mode,
        'budget': budget,
        'loss': 'margin',
        'iterations': result['iterations'],
        'success': result['success'],
        'adversarial_class': result['adversarial_class'],
        'switch_iteration': result['switch_iteration'] if result['switch_iteration'] is not None else '',
        'locked_class': result['locked_class'] if result['locked_class'] is not None else '',
        'timestamp': datetime.now().isoformat(),
    }


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Margin-loss ablation benchmark (SquareAttack, ResNet-50)"
    )
    parser.add_argument('--part', type=int, required=True, choices=[1, 2],
                        help="Part 1 = first half, Part 2 = second half")
    parser.add_argument('--clear', action='store_true',
                        help="Delete previous CSV results before running")
    parser.add_argument('--n-images', type=int, default=100,
                        help="Number of images to use (default: 100)")
    parser.add_argument('--image-seed', type=int, default=42,
                        help="Seed for image selection (default: 42)")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Device: {device}")
    print(f"Model: {MODEL_NAME} ({SOURCE})")
    print(f"Loss: margin")
    print(f"Epsilon: {EPSILON:.6f} ({EPSILON * 255:.0f}/255)")
    print(f"Budget: {MAX_BUDGET}")
    print(f"Stability threshold: {STABILITY_THRESHOLD}")

    RESULTS_DIR.mkdir(exist_ok=True)
    csv_path = CSV_PATH

    if args.clear and csv_path.exists():
        csv_path.unlink()
        print("Cleared previous results")

    existing_keys = load_existing_keys(csv_path)

    # Select and split images
    all_images = select_images(VAL_DIR, args.n_images, args.image_seed)
    half = args.n_images // 2
    if args.part == 1:
        image_paths = all_images[:half]
    else:
        image_paths = all_images[half:]

    n_images = len(image_paths)
    total_runs = n_images * 2  # untargeted + opportunistic
    print(f"Images: {n_images} (part {args.part} of {args.n_images}, seed={args.image_seed})")
    print(f"Total runs: {total_runs}")
    print()

    # Load model
    print(f"Loading model: {MODEL_NAME} ({SOURCE})...")
    model = load_benchmark_model(MODEL_NAME, SOURCE, device)

    # Preload images
    images = []
    for path in image_paths:
        x = load_benchmark_image(path, device)
        y_true = get_true_label(model, x)
        image_name = path.name
        images.append((image_name, x, y_true))
        print(f"  {image_name}: true_label={y_true}")

    completed = 0
    success_count = 0
    start_time = time.time()

    # ------------------------------------------------------------------
    # Phase 1: Margin untargeted
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"Phase 1: Margin untargeted ({n_images} images)")
    print(f"{'='*70}")

    for image_name, x, y_true in images:
        y_true_tensor = torch.tensor([y_true], device=device)
        key = (image_name, 'untargeted')
        if key in existing_keys:
            continue

        result = run_attack(model, x, y_true_tensor, 'untargeted',
                            MAX_BUDGET, device)
        row = make_row(image_name, y_true, 'untargeted', MAX_BUDGET, result)
        append_row(row, csv_path)

        completed += 1
        if result['success']:
            success_count += 1
        status = 'OK' if result['success'] else 'FAIL'
        print(f"[{completed}/{total_runs}] untargeted | {image_name} | "
              f"{result['iterations']} iters | {status}")

    # ------------------------------------------------------------------
    # Phase 2: Margin + OT (opportunistic)
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"Phase 2: Margin + OT ({n_images} images)")
    print(f"{'='*70}")

    for image_name, x, y_true in images:
        y_true_tensor = torch.tensor([y_true], device=device)
        key = (image_name, 'opportunistic')
        if key in existing_keys:
            continue

        result = run_attack(model, x, y_true_tensor, 'opportunistic',
                            MAX_BUDGET, device)
        row = make_row(image_name, y_true, 'opportunistic', MAX_BUDGET, result)
        append_row(row, csv_path)

        completed += 1
        if result['success']:
            success_count += 1
        status = 'OK' if result['success'] else 'FAIL'
        extra = ''
        if result['switch_iteration'] is not None:
            extra = (f" (switch@{result['switch_iteration']}, "
                     f"locked={result['locked_class']})")
        print(f"[{completed}/{total_runs}] opportunistic | {image_name} | "
              f"{result['iterations']} iters | {status}{extra}")

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Benchmark complete in {elapsed:.0f}s")
    print(f"Results: {csv_path}")
    print(f"Completed: {completed} runs "
          f"({success_count} successes, "
          f"{100*success_count/max(completed,1):.1f}%)")


if __name__ == '__main__':
    main()
