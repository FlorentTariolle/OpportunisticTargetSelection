"""Stability-threshold ablation: sweep S on opportunistic SimBA & SquareAttack.

Runs opportunistic SimBA and SquareAttack (CE loss) for S in {2, 3, 5, 8, 10}
on N images.  Reuses S=5 results from benchmark_winrate.csv when available,
and looks up oracle targets from the same file to avoid redundant probes.

Use --method to run a single attack method, so you can launch two terminals
in parallel (one per method) for ~2x throughput.

Usage:
    python benchmark_ablation_s.py                          # Both methods
    python benchmark_ablation_s.py --method SimBA           # Terminal 1
    python benchmark_ablation_s.py --method SquareAttack    # Terminal 2
    python benchmark_ablation_s.py --n-images 2             # Smoke test
    python benchmark_ablation_s.py --clear                  # Clear previous CSV
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
MODEL_NAME = 'resnet50'
SOURCE = 'standard'
EPSILON = 8 / 255
METHODS = ['SimBA', 'SquareAttack']
S_VALUES = {
    'SimBA': [2, 3, 5, 8, 10, 12, 15],
    'SquareAttack': [2, 3, 5, 8, 10, 12, 15],
}
VAL_DIR = Path('data/imagenet/val')
RESULTS_DIR = Path('results')
CSV_PATH = RESULTS_DIR / 'benchmark_ablation_s.csv'
WINRATE_CSV = RESULTS_DIR / 'benchmark_winrate.csv'

CSV_COLUMNS = [
    'method', 's_value', 'image', 'true_label', 'iterations', 'success',
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
    """Load existing (method, s_value, image) keys for resume."""
    keys = set()
    if not path.exists():
        return keys
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Backwards compat: old CSV rows without 'method' are SimBA
            method = row.get('method', 'SimBA')
            keys.add((method, row['s_value'], row['image']))
    return keys


def import_s5_from_winrate(winrate_csv: Path, image_names: set,
                           ablation_csv: Path, existing_keys: set) -> int:
    """Import S=5 opportunistic rows from benchmark_winrate.csv.

    Imports both SimBA and SquareAttack opportunistic rows (S=5 is the
    stability threshold used in the winrate benchmark).

    Returns number of rows imported.
    """
    if not winrate_csv.exists():
        return 0
    imported = 0
    with open(winrate_csv, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row['method']
            if (method not in ('SimBA', 'SquareAttack')
                    or row['mode'] != 'opportunistic'
                    or row['image'] not in image_names):
                continue
            key = (method, '5', row['image'])
            if key in existing_keys:
                continue
            ablation_row = {
                'method': method,
                's_value': 5,
                'image': row['image'],
                'true_label': row['true_label'],
                'iterations': row['iterations'],
                'success': row['success'],
                'adversarial_class': row['adversarial_class'],
                'switch_iteration': row.get('switch_iteration', ''),
                'locked_class': row.get('locked_class', ''),
                'timestamp': row.get('timestamp', datetime.now().isoformat()),
            }
            append_row(ablation_row, ablation_csv)
            existing_keys.add(key)
            imported += 1
    return imported


# ===========================================================================
# Attack runner
# ===========================================================================
def _run_opportunistic(model, method, x, y_true_tensor, s_val, budget, device):
    """Run a single opportunistic attack with the given S value."""
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
        description="Stability-threshold ablation for opportunistic attacks"
    )
    parser.add_argument('--clear', action='store_true',
                        help="Delete previous CSV results before running")
    parser.add_argument('--n-images', type=int, default=100,
                        help="Number of images (default: 100)")
    parser.add_argument('--budget', type=int, default=15_000,
                        help="Query budget per run (default: 15000)")
    parser.add_argument('--image-seed', type=int, default=42,
                        help="Seed for image selection (default: 42)")
    parser.add_argument('--method', type=str, default=None,
                        choices=['SimBA', 'SquareAttack'],
                        help="Run only this method (for parallel terminals)")
    args = parser.parse_args()

    methods = [args.method] if args.method else METHODS

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Device: {device}")
    print(f"Model: {MODEL_NAME} ({SOURCE})")
    print(f"Epsilon: {EPSILON:.6f} ({EPSILON * 255:.0f}/255)")
    print(f"Methods: {methods}")
    print(f"S values: { {m: S_VALUES[m] for m in methods} }")
    print(f"Images: {args.n_images} (seed={args.image_seed})")
    print(f"Budget: {args.budget}")
    print()

    RESULTS_DIR.mkdir(exist_ok=True)

    if args.clear and CSV_PATH.exists():
        CSV_PATH.unlink()
        print("Cleared previous results")

    existing_keys = load_existing_keys(CSV_PATH)
    if existing_keys:
        print(f"Resuming: found {len(existing_keys)} existing results")

    # Load model and images
    print(f"\nLoading model: {MODEL_NAME} ({SOURCE})...")
    model = load_benchmark_model(MODEL_NAME, SOURCE, device)

    print(f"Selecting {args.n_images} images from {VAL_DIR}...")
    image_paths = select_images(VAL_DIR, args.n_images, args.image_seed)

    images = []
    image_name_set = set()
    for path in image_paths:
        x = load_benchmark_image(path, device)
        y_true = get_true_label(model, x)
        name = path.name
        images.append((name, x, y_true))
        image_name_set.add(name)
        print(f"  {name}: true_label={y_true}")

    # Import S=5 from winrate benchmark (both methods)
    imported = import_s5_from_winrate(
        WINRATE_CSV, image_name_set, CSV_PATH, existing_keys)
    if imported:
        print(f"\nImported {imported} S=5 rows from {WINRATE_CSV}")
        existing_keys = load_existing_keys(CSV_PATH)

    # Build work queue
    jobs = []
    for method in methods:
        for s_val in S_VALUES[method]:
            for image_name, x, y_true in images:
                key = (method, str(s_val), image_name)
                if key not in existing_keys:
                    jobs.append((method, s_val, image_name, x, y_true))

    total_runs = sum(len(S_VALUES[m]) for m in methods) * len(images)
    completed = total_runs - len(jobs)
    start_time = time.time()

    print(f"\n{'='*70}")
    print(f"Running ablation: {len(methods)} methods x "
          f"{len(images)} images")
    print(f"Total runs: {total_runs} | pending: {len(jobs)}")
    print(f"{'='*70}")

    if not jobs:
        print("Nothing to do.")
        return

    for method, s_val, image_name, x, y_true in jobs:
        y_true_tensor = torch.tensor([y_true], device=device)
        result = _run_opportunistic(
            model, method, x, y_true_tensor, s_val, args.budget, device)

        row = {
            'method': method,
            's_value': s_val,
            'image': image_name,
            'true_label': y_true,
            'iterations': result['iterations'],
            'success': result['success'],
            'adversarial_class': result['adversarial_class'],
            'switch_iteration': result['switch_iteration'] if result['switch_iteration'] is not None else '',
            'locked_class': result['locked_class'] if result['locked_class'] is not None else '',
            'timestamp': datetime.now().isoformat(),
        }
        append_row(row, CSV_PATH)
        completed += 1

        status = 'OK' if result['success'] else 'FAIL'
        extra = ''
        if result['switch_iteration'] is not None:
            extra = (f" (switch@{result['switch_iteration']}, "
                     f"locked={result['locked_class']})")
        print(f"[{completed}/{total_runs}] {method} S={s_val} | "
              f"{image_name} | {result['iterations']} iters | "
              f"{status}{extra}")

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Ablation complete in {elapsed:.0f}s")
    print(f"Results: {CSV_PATH}")
    print(f"Completed: {completed} runs")


if __name__ == '__main__':
    main()
