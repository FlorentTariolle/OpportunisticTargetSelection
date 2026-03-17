"""Naive fixed-iteration switching ablation.

Compares OT's stability heuristic against a naive baseline that switches to
targeted at a fixed iteration T (no stability check).

Usage:
    python benchmark_ablation_naive.py                                    # Standard (T sweep)
    python benchmark_ablation_naive.py --source robust                   # Robust (T=100 vs OT)
    python benchmark_ablation_naive.py --image-start 0 --image-end 50    # Terminal 1
    python benchmark_ablation_naive.py --image-start 50 --image-end 100  # Terminal 2
    python benchmark_ablation_naive.py --n-images 2                      # Smoke test
    python benchmark_ablation_naive.py --clear                           # Clear CSV
"""

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import torch

from benchmarks.benchmark import load_benchmark_model, load_benchmark_image, get_true_label
from benchmarks.winrate import select_images
from src.attacks.simba import SimBA
from src.attacks.square import SquareAttack

# ===========================================================================
# Configuration
# ===========================================================================
MODEL_NAMES = {'standard': 'resnet50', 'robust': 'Salman2020Do_R50'}
EPSILON = 8 / 255
METHODS = ['SimBA', 'SquareAttack']
T_VALUES_STANDARD = [5, 10, 20, 50, 100, 200, 500]
T_VALUES_ROBUST = [100]
# Default S values for OT baseline (must match benchmark_winrate.py)
OT_S = {'SimBA': 10, 'SquareAttack': 8}
VAL_DIR = Path('data/imagenet/val')
RESULTS_DIR = Path('results')

CSV_COLUMNS = [
    'method', 't_value', 'image', 'true_label', 'iterations', 'success',
    'adversarial_class', 'switch_iteration', 'locked_class', 'timestamp',
]


# ===========================================================================
# CSV I/O
# ===========================================================================
def append_row(row: dict, path: Path):
    file_exists = path.exists() and path.stat().st_size > 0
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def load_existing_keys(path: Path) -> set:
    """Load existing (method, t_value, image) keys for resume."""
    keys = set()
    if not path.exists():
        return keys
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            keys.add((row['method'], row['t_value'], row['image']))
    return keys


# ===========================================================================
# Attack runners
# ===========================================================================
def _run_naive(model, method, x, y_true_tensor, t_val, budget, device):
    """Run opportunistic attack with naive switching at iteration T."""
    y_true = y_true_tensor.item()

    if method == 'SimBA':
        attack = SimBA(
            model=model, epsilon=EPSILON, max_iterations=budget,
            device=device, use_dct=True, pixel_range=(0.0, 1.0),
        )
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(0)
    else:
        attack = SquareAttack(
            model=model, epsilon=EPSILON, max_iterations=budget,
            device=device, loss='ce', normalize=False, seed=0,
        )

    x_adv = attack.generate(
        x, y_true_tensor,
        track_confidence=True,
        targeted=False,
        early_stop=True,
        opportunistic=True,
        stability_threshold=OT_S[method],
        naive_switch_iteration=t_val,
    )

    conf_hist = attack.confidence_history
    if conf_hist and conf_hist.get('iterations'):
        iterations = conf_hist['iterations'][-1]
    else:
        iterations = budget

    with torch.no_grad():
        logits = model(x_adv)
        pred = logits.argmax(dim=1).item()
    success = (pred != y_true)

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


def _run_ot_baseline(model, method, x, y_true_tensor, budget, device):
    """Run OT with default S as baseline."""
    y_true = y_true_tensor.item()
    s_val = OT_S[method]

    if method == 'SimBA':
        attack = SimBA(
            model=model, epsilon=EPSILON, max_iterations=budget,
            device=device, use_dct=True, pixel_range=(0.0, 1.0),
        )
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(0)
    else:
        attack = SquareAttack(
            model=model, epsilon=EPSILON, max_iterations=budget,
            device=device, loss='ce', normalize=False, seed=0,
        )

    x_adv = attack.generate(
        x, y_true_tensor,
        track_confidence=True,
        targeted=False,
        early_stop=True,
        opportunistic=True,
        stability_threshold=s_val,
    )

    conf_hist = attack.confidence_history
    if conf_hist and conf_hist.get('iterations'):
        iterations = conf_hist['iterations'][-1]
    else:
        iterations = budget

    with torch.no_grad():
        logits = model(x_adv)
        pred = logits.argmax(dim=1).item()
    success = (pred != y_true)

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


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Naive fixed-iteration switching ablation"
    )
    parser.add_argument('--clear', action='store_true',
                        help="Delete previous CSV results before running")
    parser.add_argument('--n-images', type=int, default=100,
                        help="Number of images (default: 100)")
    parser.add_argument('--budget', type=int, default=15_000,
                        help="Query budget per run (default: 15000)")
    parser.add_argument('--image-seed', type=int, default=42,
                        help="Seed for image selection (default: 42)")
    parser.add_argument('--image-start', type=int, default=0,
                        help="Start index for image slice (default: 0)")
    parser.add_argument('--image-end', type=int, default=None,
                        help="End index for image slice (default: all)")
    parser.add_argument('--source', type=str, default='standard',
                        choices=['standard', 'robust'],
                        help="Model source (default: standard)")
    args = parser.parse_args()

    source = args.source
    model_name = MODEL_NAMES[source]
    t_values = T_VALUES_STANDARD if source == 'standard' else T_VALUES_ROBUST
    csv_path = RESULTS_DIR / f'benchmark_ablation_naive_{source}.csv'
    methods = METHODS

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Device: {device}")
    print(f"Model: {model_name} ({source})")
    print(f"Epsilon: {EPSILON:.6f} ({EPSILON * 255:.0f}/255)")
    print(f"Methods: {METHODS}")
    print(f"T values: {t_values}")
    print(f"OT baseline S: {OT_S}")
    print(f"Images: {args.n_images} (seed={args.image_seed}), "
          f"slice [{args.image_start}:{args.image_end}]")
    print(f"Budget: {args.budget}")
    print()

    RESULTS_DIR.mkdir(exist_ok=True)

    if args.clear and csv_path.exists():
        csv_path.unlink()
        print("Cleared previous results")

    existing_keys = load_existing_keys(csv_path)
    if existing_keys:
        print(f"Resuming: found {len(existing_keys)} existing results")

    # Load model and images
    # A100 optimizations: TF32 matmul + cuDNN autotuning
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    print(f"\nLoading model: {model_name} ({source})...")
    model = load_benchmark_model(model_name, source, device)

    print(f"Selecting {args.n_images} images from {VAL_DIR}...")
    image_paths = select_images(VAL_DIR, args.n_images, args.image_seed)
    image_paths = image_paths[args.image_start:args.image_end]
    print(f"Slice: [{args.image_start}:{args.image_end}] -> {len(image_paths)} images")

    images = []
    for path in image_paths:
        x = load_benchmark_image(path, device)
        y_true = get_true_label(model, x)
        name = path.name
        images.append((name, x, y_true))
        print(f"  {name}: true_label={y_true}")

    # Build work queue: T values + OT baseline
    jobs = []
    for method in methods:
        for t_val in t_values:
            for image_name, x, y_true in images:
                key = (method, str(t_val), image_name)
                if key not in existing_keys:
                    jobs.append((method, str(t_val), t_val, image_name, x, y_true))
        # OT baseline
        for image_name, x, y_true in images:
            key = (method, 'OT', image_name)
            if key not in existing_keys:
                jobs.append((method, 'OT', None, image_name, x, y_true))

    total_runs = len(methods) * (len(t_values) + 1) * len(images)
    completed = total_runs - len(jobs)
    start_time = time.time()

    print(f"\n{'='*70}")
    print(f"Running naive switching ablation: {len(methods)} methods x "
          f"{len(images)} images")
    print(f"Total runs: {total_runs} | pending: {len(jobs)}")
    print(f"{'='*70}")

    if not jobs:
        print("Nothing to do.")
        return

    for method, t_label, t_val, image_name, x, y_true in jobs:
        y_true_tensor = torch.tensor([y_true], device=device)

        if t_label == 'OT':
            result = _run_ot_baseline(
                model, method, x, y_true_tensor, args.budget, device)
        else:
            result = _run_naive(
                model, method, x, y_true_tensor, t_val, args.budget, device)

        row = {
            'method': method,
            't_value': t_label,
            'image': image_name,
            'true_label': y_true,
            'iterations': result['iterations'],
            'success': result['success'],
            'adversarial_class': result['adversarial_class'],
            'switch_iteration': result['switch_iteration'] if result['switch_iteration'] is not None else '',
            'locked_class': result['locked_class'] if result['locked_class'] is not None else '',
            'timestamp': datetime.now().isoformat(),
        }
        append_row(row, csv_path)
        completed += 1

        run_time = time.time() - start_time
        avg_time = run_time / (completed - (total_runs - len(jobs)))
        eta = avg_time * (total_runs - completed)
        status = 'OK' if result['success'] else 'FAIL'
        extra = ''
        if result['switch_iteration'] is not None:
            extra = (f" (switch@{result['switch_iteration']}, "
                     f"locked={result['locked_class']})")
        print(f"[{completed}/{total_runs}] {method} T={t_label} | "
              f"{image_name} | {result['iterations']} iters | "
              f"{status}{extra} | "
              f"{run_time:.0f}s elapsed, ETA {eta:.0f}s")

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Naive ablation complete in {elapsed:.0f}s")
    print(f"Results: {csv_path}")
    print(f"Completed: {completed} runs")


if __name__ == '__main__':
    main()
