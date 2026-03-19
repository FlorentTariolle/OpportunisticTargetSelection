"""Multi-seed validation benchmark (#24).

Runs SimBA and SquareAttack (CE) on ResNet-50 with 5 attack seeds to validate
that OT's effect is not seed-dependent.

Each (method, image, mode, seed) combination is one run.
Total: 2 methods × 100 images × 3 modes × 5 seeds = 3,000 runs.

Usage:
    python benchmarks/multiseed.py --image-start 0 --image-end 100
    python benchmarks/multiseed.py --image-start 0 --image-end 10   # smoke test
    python benchmarks/multiseed.py --clear                          # clear CSV
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

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
BUDGET = 15_000
SEEDS = [0, 1, 2, 3, 4]
METHODS = ['SimBA', 'SquareAttack']
MODES = ['untargeted', 'targeted', 'opportunistic']
STABILITY_THRESHOLD = {'SimBA': 10, 'SquareAttack': 8}
VAL_DIR = Path('data/imagenet/val')
RESULTS_DIR = Path('results')
CSV_PATH = RESULTS_DIR / 'benchmark_multiseed.csv'

CSV_COLUMNS = [
    'method', 'image', 'true_label', 'mode', 'seed',
    'iterations', 'success', 'adversarial_class', 'oracle_target',
    'switch_iteration', 'locked_class', 'timestamp',
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
    keys = set()
    if not path.exists():
        return keys
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            keys.add((row['method'], row['image'], row['mode'], row['seed']))
    return keys


# ===========================================================================
# Attack runner
# ===========================================================================
def run_attack(model, method, x, y_true_tensor, mode, target_class,
               seed, device):
    """Run a single attack. Returns result dict."""
    y_true_int = y_true_tensor.item()
    is_targeted = (mode == 'targeted')
    is_opportunistic = (mode == 'opportunistic')
    target_tensor = None
    if is_targeted and target_class is not None:
        target_tensor = torch.tensor([target_class], device=device)

    if method == 'SimBA':
        attack = SimBA(
            model=model, epsilon=EPSILON, max_iterations=BUDGET,
            device=device, use_dct=True, pixel_range=(0.0, 1.0),
        )
    else:
        attack = SquareAttack(
            model=model, epsilon=EPSILON, max_iterations=BUDGET,
            device=device, loss='ce', normalize=False, seed=seed,
        )

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    x_adv = attack.generate(
        x, y_true_tensor,
        track_confidence=True,
        targeted=is_targeted,
        target_class=target_tensor,
        early_stop=True,
        opportunistic=is_opportunistic,
        stability_threshold=STABILITY_THRESHOLD[method],
    )

    conf_hist = attack.confidence_history
    if conf_hist and conf_hist.get('iterations'):
        iterations = conf_hist['iterations'][-1]
    else:
        iterations = BUDGET

    with torch.no_grad():
        pred = model(x_adv).argmax(dim=1).item()

    if is_targeted:
        success = (pred == target_class)
    else:
        success = (pred != y_true_int)

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


def get_oracle_target(model, method, x, y_true_tensor, seed, device):
    """Run untargeted to determine oracle target for a given seed."""
    result = run_attack(model, method, x, y_true_tensor, 'untargeted',
                        None, seed, device)
    y_true_int = y_true_tensor.item()
    if result['success']:
        return result['adversarial_class']
    # Fallback: argmax non-true class on clean image
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)
        probs[0][y_true_int] = -1.0
        return probs.argmax(dim=1).item()


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Multi-seed validation benchmark (#24)")
    parser.add_argument('--clear', action='store_true')
    parser.add_argument('--n-images', type=int, default=100)
    parser.add_argument('--image-seed', type=int, default=42)
    parser.add_argument('--image-start', type=int, default=0)
    parser.add_argument('--image-end', type=int, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Device: {device}")
    print(f"Model: {MODEL_NAME} ({SOURCE})")
    print(f"Epsilon: {EPSILON:.6f} ({EPSILON * 255:.0f}/255)")
    print(f"Budget: {BUDGET}")
    print(f"Seeds: {SEEDS}")
    print(f"Methods: {METHODS}")
    print()

    RESULTS_DIR.mkdir(exist_ok=True)

    if args.clear and CSV_PATH.exists():
        CSV_PATH.unlink()
        print("Cleared previous results")

    existing_keys = load_existing_keys(CSV_PATH)
    if existing_keys:
        print(f"Resuming: found {len(existing_keys)} existing results")

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    print(f"Loading model: {MODEL_NAME}...")
    model = load_benchmark_model(MODEL_NAME, SOURCE, device)

    print(f"Selecting {args.n_images} images from {VAL_DIR}...")
    image_paths = select_images(VAL_DIR, args.n_images, args.image_seed)
    image_paths = image_paths[args.image_start:args.image_end]
    print(f"Slice: [{args.image_start}:{args.image_end}] -> "
          f"{len(image_paths)} images")

    images = []
    for path in image_paths:
        x = load_benchmark_image(path, device)
        y_true = get_true_label(model, x)
        images.append((path.name, x, y_true))

    # Build work queue: (method, image_name, x, y_true, seed)
    # Order: iterate seeds inside images so oracle cache stays warm
    jobs = []
    for method in METHODS:
        for image_name, x, y_true in images:
            for seed in SEEDS:
                for mode in MODES:
                    key = (method, image_name, mode, str(seed))
                    if key not in existing_keys:
                        jobs.append((method, image_name, x, y_true, seed, mode))

    total_runs = len(METHODS) * len(images) * len(SEEDS) * len(MODES)
    completed = total_runs - len(jobs)
    start_time = time.time()

    print(f"\n{'='*70}")
    print(f"Multi-seed benchmark: {len(METHODS)} methods x "
          f"{len(images)} images x {len(SEEDS)} seeds x {len(MODES)} modes")
    print(f"Total runs: {total_runs} | pending: {len(jobs)}")
    print(f"{'='*70}")

    if not jobs:
        print("Nothing to do.")
        return

    # Pre-load oracle targets from existing CSV (for resume)
    oracle_cache = {}
    if CSV_PATH.exists():
        with open(CSV_PATH, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['mode'] == 'untargeted' and row['success'].lower() == 'true':
                    cache_key = (row['method'], row['image'], int(row['seed']))
                    oracle_cache[cache_key] = int(float(row['adversarial_class']))
                elif row['mode'] in ('targeted', 'opportunistic'):
                    ot = row.get('oracle_target', '')
                    if ot and ot != '':
                        cache_key = (row['method'], row['image'], int(row['seed']))
                        oracle_cache.setdefault(cache_key, int(float(ot)))

    for method, image_name, x, y_true, seed, mode in jobs:
        y_true_tensor = torch.tensor([y_true], device=device)

        oracle_target = None
        if mode == 'targeted':
            cache_key = (method, image_name, seed)
            if cache_key not in oracle_cache:
                oracle_cache[cache_key] = get_oracle_target(
                    model, method, x, y_true_tensor, seed, device)
            oracle_target = oracle_cache[cache_key]

        result = run_attack(model, method, x, y_true_tensor, mode,
                            oracle_target, seed, device)

        # Cache oracle from untargeted runs
        if mode == 'untargeted' and result['success']:
            oracle_cache[(method, image_name, seed)] = result['adversarial_class']

        row = {
            'method': method,
            'image': image_name,
            'true_label': y_true,
            'mode': mode,
            'seed': seed,
            'iterations': result['iterations'],
            'success': result['success'],
            'adversarial_class': result['adversarial_class'],
            'oracle_target': oracle_target if oracle_target is not None else '',
            'switch_iteration': result['switch_iteration'] if result['switch_iteration'] is not None else '',
            'locked_class': result['locked_class'] if result['locked_class'] is not None else '',
            'timestamp': datetime.now().isoformat(),
        }
        append_row(row, CSV_PATH)
        completed += 1

        elapsed = time.time() - start_time
        done_this = completed - (total_runs - len(jobs))
        avg = elapsed / done_this if done_this > 0 else 0
        eta = avg * (total_runs - completed)
        status = 'OK' if result['success'] else 'FAIL'
        extra = ''
        if mode == 'targeted':
            extra = f" target={oracle_target}"
        if mode == 'opportunistic' and result['switch_iteration'] is not None:
            extra = f" switch@{result['switch_iteration']}"

        print(f"[{completed}/{total_runs}] {method} {mode} seed={seed} | "
              f"{image_name} | {result['iterations']} iters | {status}{extra} | "
              f"{elapsed:.0f}s, ETA {eta:.0f}s")

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Multi-seed benchmark complete in {elapsed:.0f}s")
    print(f"Results: {CSV_PATH}")
    print(f"Completed: {completed} runs")


if __name__ == '__main__':
    main()
