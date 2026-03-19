"""Target-selection ablation: T=0 (clean argmax) and random-target baselines.

Tests whether OT's value comes from target *selection* or just from targeting.

New conditions (run by this script):
  clean_argmax  — target = argmax non-true class on the clean image (T=0)
  random_target — target = uniform random non-true class

Reference baselines (from existing CSVs, not re-run):
  untargeted    — benchmark_standard.csv (ResNet-50, 10K budget)
  OT            — benchmark_ablation_naive_standard.csv (t_value=OT, 15K budget)
  oracle        — benchmark_standard.csv (ResNet-50, 10K budget)

Usage:
    python ablation_target_selection.py                                    # Full run
    python ablation_target_selection.py --n-images 2                      # Smoke test
    python ablation_target_selection.py --image-start 0 --image-end 50    # Parallel shard
    python ablation_target_selection.py --clear                           # Clear CSV
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import csv
import random
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
METHODS = ['SimBA', 'SquareAttack']
CONDITIONS = ['clean_argmax', 'random_target']
ATTACK_SEED = 0
VAL_DIR = Path('data/imagenet/val')
RESULTS_DIR = Path('results')

CSV_COLUMNS = [
    'method', 'condition', 'image', 'true_label',
    'target_class', 'iterations', 'success',
    'adversarial_class', 'timestamp',
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
            keys.add((row['method'], row['condition'], row['image']))
    return keys


# ===========================================================================
# Attack factory
# ===========================================================================
def create_attack(method: str, model, budget: int, device):
    if method == 'SimBA':
        return SimBA(
            model=model, epsilon=EPSILON, max_iterations=budget,
            device=device, use_dct=True, pixel_range=(0.0, 1.0),
        )
    else:
        return SquareAttack(
            model=model, epsilon=EPSILON, max_iterations=budget,
            device=device, loss='ce', normalize=False, seed=ATTACK_SEED,
        )


# ===========================================================================
# Target selection
# ===========================================================================
def select_clean_argmax(model, x, y_true: int) -> int:
    """T=0: argmax non-true class from clean-image logits."""
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        probs[0][y_true] = -1.0
        return probs.argmax(dim=1).item()


def select_random_target(y_true: int, n_classes: int, seed: int,
                         image_name: str) -> int:
    """Uniform random non-true class, deterministic per (seed, image)."""
    rng = random.Random((seed, image_name))
    candidates = [c for c in range(n_classes) if c != y_true]
    return rng.choice(candidates)


# ===========================================================================
# Run one condition
# ===========================================================================
def run_condition(model, method, condition, x, y_true, image_name,
                  budget, device):
    """Run a single (method, condition, image) experiment."""
    y_true_tensor = torch.tensor([y_true], device=device)
    attack = create_attack(method, model, budget, device)

    # Seed for SimBA reproducibility
    torch.manual_seed(ATTACK_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(ATTACK_SEED)

    target_class = None

    if condition == 'clean_argmax':
        target_class = select_clean_argmax(model, x, y_true)
        x_adv = attack.generate(
            x, y_true_tensor,
            track_confidence=True, targeted=True,
            target_class=torch.tensor([target_class], device=device),
            early_stop=True,
        )
    elif condition == 'random_target':
        target_class = select_random_target(y_true, 1000, ATTACK_SEED,
                                            image_name)
        x_adv = attack.generate(
            x, y_true_tensor,
            track_confidence=True, targeted=True,
            target_class=torch.tensor([target_class], device=device),
            early_stop=True,
        )
    # Extract results
    conf_hist = attack.confidence_history
    if conf_hist and conf_hist.get('iterations'):
        iterations = conf_hist['iterations'][-1]
    else:
        iterations = budget

    with torch.no_grad():
        pred = model(x_adv).argmax(dim=1).item()

    success = (pred == target_class)

    return {
        'target_class': target_class,
        'iterations': iterations,
        'success': success,
        'adversarial_class': pred,
    }


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Target-selection ablation (T=0 / random-target)")
    parser.add_argument('--clear', action='store_true')
    parser.add_argument('--n-images', type=int, default=100)
    parser.add_argument('--budget', type=int, default=15_000)
    parser.add_argument('--image-seed', type=int, default=42)
    parser.add_argument('--image-start', type=int, default=0)
    parser.add_argument('--image-end', type=int, default=None)
    args = parser.parse_args()

    csv_path = RESULTS_DIR / 'benchmark_ablation_target_selection.csv'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Device: {device}")
    print(f"Model: {MODEL_NAME}")
    print(f"Epsilon: {EPSILON:.6f} ({EPSILON * 255:.0f}/255)")
    print(f"Methods: {METHODS}")
    print(f"Conditions: {CONDITIONS}")
    print(f"Budget: {args.budget}")
    print(f"Images: {args.n_images} (seed={args.image_seed}), "
          f"slice [{args.image_start}:{args.image_end}]")
    print()

    RESULTS_DIR.mkdir(exist_ok=True)

    if args.clear and csv_path.exists():
        csv_path.unlink()
        print("Cleared previous results")

    existing_keys = load_existing_keys(csv_path)
    if existing_keys:
        print(f"Resuming: found {len(existing_keys)} existing results")

    # GPU optimizations
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

    # Preload images
    images = []
    for path in image_paths:
        x = load_benchmark_image(path, device)
        y_true = get_true_label(model, x)
        images.append((path.name, x, y_true))
        print(f"  {path.name}: true_label={y_true}")

    # Build work queue
    jobs = []
    for method in METHODS:
        for condition in CONDITIONS:
            for image_name, x, y_true in images:
                key = (method, condition, image_name)
                if key not in existing_keys:
                    jobs.append((method, condition, image_name, x, y_true))

    total_runs = len(METHODS) * len(CONDITIONS) * len(images)
    completed = total_runs - len(jobs)
    start_time = time.time()

    print(f"\n{'='*70}")
    print(f"Target-selection ablation: {len(METHODS)} methods x "
          f"{len(CONDITIONS)} conditions x {len(images)} images")
    print(f"Total runs: {total_runs} | pending: {len(jobs)}")
    print(f"{'='*70}")

    if not jobs:
        print("Nothing to do.")
        return

    for method, condition, image_name, x, y_true in jobs:
        result = run_condition(model, method, condition, x, y_true,
                               image_name, args.budget, device)

        row = {
            'method': method,
            'condition': condition,
            'image': image_name,
            'true_label': y_true,
            'target_class': result['target_class'],
            'iterations': result['iterations'],
            'success': result['success'],
            'adversarial_class': result['adversarial_class'],
            'timestamp': datetime.now().isoformat(),
        }
        append_row(row, csv_path)
        completed += 1

        elapsed = time.time() - start_time
        done_this_session = completed - (total_runs - len(jobs))
        avg_time = elapsed / done_this_session if done_this_session > 0 else 0
        eta = avg_time * (total_runs - completed)
        status = 'OK' if result['success'] else 'FAIL'
        extra = f" (target={result['target_class']})"
        print(f"[{completed}/{total_runs}] {method} {condition} | "
              f"{image_name} | {result['iterations']} iters | "
              f"{status}{extra} | "
              f"{elapsed:.0f}s elapsed, ETA {eta:.0f}s")

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Target-selection ablation complete in {elapsed:.0f}s")
    print(f"Results: {csv_path}")
    print(f"Completed: {completed} runs")


if __name__ == '__main__':
    main()
