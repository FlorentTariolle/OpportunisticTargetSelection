"""Robust S ablation: sweep stability threshold on Salman2020Do_R50.

Runs SquareAttack (CE loss) in 3 modes (untargeted, oracle targeted,
opportunistic) for S in {10, 12, 14, 16, 18, 20} on 50 images.

Split by images for parallel execution:
  --part 1  runs images 0-24
  --part 2  runs images 25-49

Usage:
    python benchmark_ablation_s_robust.py --part 1
    python benchmark_ablation_s_robust.py --part 2
    python benchmark_ablation_s_robust.py --part 1 --n-images 4  # smoke test
    python benchmark_ablation_s_robust.py --clear --part 1       # clear + run
"""

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmarks.benchmark import load_benchmark_model, load_benchmark_image, get_true_label
from benchmarks.winrate import select_images
from src.attacks.square import SquareAttack

# ===========================================================================
# Configuration
# ===========================================================================
MODEL_NAME = 'Salman2020Do_R50'
SOURCE = 'robust'
EPSILON = 8 / 255
BUDGET = 20_000
S_VALUES = [10, 12, 14, 16, 18, 20]
VAL_DIR = Path('data/imagenet/val')
RESULTS_DIR = Path('results')
CSV_PATH = RESULTS_DIR / 'benchmark_ablation_s_robust.csv'

CSV_COLUMNS = [
    's_value', 'image', 'true_label', 'mode', 'iterations', 'success',
    'adversarial_class', 'oracle_target', 'switch_iteration', 'locked_class',
    'final_margin', 'timestamp',
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
    """Load existing (mode, s_value, image) keys for resume."""
    keys = set()
    if not path.exists():
        return keys
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            keys.add((row['mode'], row['s_value'], row['image']))
    return keys


def load_oracle_targets(path: Path) -> dict:
    """Load oracle targets from CSV: {image: int_class}."""
    oracles = {}
    if not path.exists():
        return oracles
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img = row['image']
            if img in oracles:
                continue
            # Successful untargeted → adversarial_class is oracle
            if row['mode'] == 'untargeted' and row['success'] == 'True':
                oracles[img] = int(row['adversarial_class'])
            # Or pick up from oracle_target column
            elif row.get('oracle_target') and row['oracle_target'] != '':
                oracles[img] = int(row['oracle_target'])
    return oracles


# ===========================================================================
# Attack runner
# ===========================================================================
def compute_margin(model, x_adv, y_true, success):
    """Compute final margin: 0 if success, else max(P(y) - max P(k!=y), 0)."""
    if success:
        return 0.0
    with torch.no_grad():
        logits = model(x_adv)
        probs = F.softmax(logits, dim=1)
        true_conf = probs[0][y_true].item()
        probs[0][y_true] = -1.0
        max_other = probs.max(dim=1).values.item()
    return max(true_conf - max_other, 0.0)


def run_attack(model, x, y_true_tensor, mode, target_class, s_val, device):
    """Run a single SquareAttack and return result dict."""
    y_true = y_true_tensor.item()

    is_targeted = (mode == 'targeted')
    is_opportunistic = (mode == 'opportunistic')

    attack = SquareAttack(
        model=model, epsilon=EPSILON, max_iterations=BUDGET,
        device=device, loss='ce', normalize=False, seed=0,
    )

    target_tensor = None
    if is_targeted and target_class is not None:
        target_tensor = torch.tensor([target_class], device=device)

    x_adv = attack.generate(
        x, y_true_tensor,
        track_confidence=True,
        targeted=is_targeted,
        target_class=target_tensor,
        early_stop=True,
        opportunistic=is_opportunistic,
        stability_threshold=s_val if is_opportunistic else None,
    )

    conf_hist = attack.confidence_history
    if conf_hist and conf_hist.get('iterations'):
        iterations = conf_hist['iterations'][-1]
    else:
        iterations = BUDGET

    with torch.no_grad():
        logits = model(x_adv)
        pred = logits.argmax(dim=1).item()

    if is_targeted:
        success = (pred == target_class)
    else:
        success = (pred != y_true)

    margin = compute_margin(model, x_adv, y_true, success)

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
        'final_margin': margin,
    }


def determine_oracle_target(model, x, y_true_tensor, device):
    """Run untargeted attack to determine oracle target class."""
    result = run_attack(model, x, y_true_tensor, 'untargeted', None, None, device)
    if result['success']:
        return result['adversarial_class']
    # Fallback: most-predicted non-true class on clean image
    y_true = y_true_tensor.item()
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        probs[0][y_true] = -1.0
        return probs.argmax(dim=1).item()


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Robust S ablation: SquareAttack CE on Salman2020Do_R50"
    )
    parser.add_argument('--part', type=int, required=True, choices=[1, 2],
                        help="Part 1 = images 0-24, Part 2 = images 25-49")
    parser.add_argument('--clear', action='store_true',
                        help="Delete previous CSV results before running")
    parser.add_argument('--n-images', type=int, default=50,
                        help="Total images to select (default: 50)")
    parser.add_argument('--image-seed', type=int, default=42,
                        help="Seed for image selection (default: 42)")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Image split
    half = args.n_images // 2
    if args.part == 1:
        img_start, img_end = 0, half
    else:
        img_start, img_end = half, args.n_images

    print(f"Device: {device}")
    print(f"Model: {MODEL_NAME} ({SOURCE})")
    print(f"Budget: {BUDGET}")
    print(f"S values: {S_VALUES}")
    print(f"Part {args.part}: images [{img_start}, {img_end})")
    print()

    RESULTS_DIR.mkdir(exist_ok=True)

    if args.clear and CSV_PATH.exists():
        CSV_PATH.unlink()
        print("Cleared previous results")

    existing_keys = load_existing_keys(CSV_PATH)
    oracle_cache = load_oracle_targets(CSV_PATH)

    if existing_keys:
        print(f"Resuming: found {len(existing_keys)} existing results")

    # Load model and images
    print(f"\nLoading model: {MODEL_NAME} ({SOURCE})...")
    model = load_benchmark_model(MODEL_NAME, SOURCE, device)

    print(f"Selecting {args.n_images} images from {VAL_DIR}...")
    all_image_paths = select_images(VAL_DIR, args.n_images, args.image_seed)
    part_image_paths = all_image_paths[img_start:img_end]

    images = []
    for path in part_image_paths:
        x = load_benchmark_image(path, device)
        y_true = get_true_label(model, x)
        name = path.name
        images.append((name, x, y_true))
        print(f"  {name}: true_label={y_true}")

    # Per image: 1 untargeted + 1 targeted + 6 opportunistic = 8
    runs_per_image = 2 + len(S_VALUES)
    total_runs = len(images) * runs_per_image
    completed = 0
    start_time = time.time()

    # Count already-done runs for this part's images
    part_image_names = {name for name, _, _ in images}
    for key in existing_keys:
        mode, s_val, img = key
        if img in part_image_names:
            completed += 1

    print(f"\n{'='*70}")
    print(f"Part {args.part}: {len(images)} images x {runs_per_image} runs/image")
    print(f"Total: {total_runs} | completed: {completed} | pending: {total_runs - completed}")
    print(f"{'='*70}")

    if completed >= total_runs:
        print("Nothing to do.")
        return

    for image_name, x, y_true in images:
        y_true_tensor = torch.tensor([y_true], device=device)

        # --- Phase 1: Untargeted (oracle probe) ---
        key = ('untargeted', '', image_name)
        if key not in existing_keys:
            print(f"\n[untargeted] {image_name}...", end=' ', flush=True)
            result = run_attack(model, x, y_true_tensor, 'untargeted',
                                None, None, device)
            row = {
                's_value': '',
                'image': image_name,
                'true_label': y_true,
                'mode': 'untargeted',
                'iterations': result['iterations'],
                'success': result['success'],
                'adversarial_class': result['adversarial_class'],
                'oracle_target': '',
                'switch_iteration': '',
                'locked_class': '',
                'final_margin': round(result['final_margin'], 6),
                'timestamp': datetime.now().isoformat(),
            }
            append_row(row, CSV_PATH)
            existing_keys.add(key)
            completed += 1

            if result['success']:
                oracle_cache[image_name] = result['adversarial_class']
            status = 'OK' if result['success'] else 'FAIL'
            print(f"{result['iterations']} iters | {status} | "
                  f"margin={result['final_margin']:.4f}")

        # Determine oracle target
        if image_name not in oracle_cache:
            print(f"  Determining oracle for {image_name}...")
            oracle = determine_oracle_target(model, x, y_true_tensor, device)
            oracle_cache[image_name] = oracle
        oracle_target = oracle_cache[image_name]

        # --- Phase 2: Oracle targeted ---
        key = ('targeted', '', image_name)
        if key not in existing_keys:
            print(f"[targeted] {image_name} -> class {oracle_target}...",
                  end=' ', flush=True)
            result = run_attack(model, x, y_true_tensor, 'targeted',
                                oracle_target, None, device)
            row = {
                's_value': '',
                'image': image_name,
                'true_label': y_true,
                'mode': 'targeted',
                'iterations': result['iterations'],
                'success': result['success'],
                'adversarial_class': result['adversarial_class'],
                'oracle_target': oracle_target,
                'switch_iteration': '',
                'locked_class': '',
                'final_margin': round(result['final_margin'], 6),
                'timestamp': datetime.now().isoformat(),
            }
            append_row(row, CSV_PATH)
            existing_keys.add(key)
            completed += 1

            status = 'OK' if result['success'] else 'FAIL'
            print(f"{result['iterations']} iters | {status} | "
                  f"margin={result['final_margin']:.4f}")

        # --- Phase 3: Opportunistic per S ---
        for s_val in S_VALUES:
            key = ('opportunistic', str(s_val), image_name)
            if key in existing_keys:
                continue

            print(f"[S={s_val}] {image_name}...", end=' ', flush=True)
            result = run_attack(model, x, y_true_tensor, 'opportunistic',
                                None, s_val, device)
            row = {
                's_value': s_val,
                'image': image_name,
                'true_label': y_true,
                'mode': 'opportunistic',
                'iterations': result['iterations'],
                'success': result['success'],
                'adversarial_class': result['adversarial_class'],
                'oracle_target': oracle_target,
                'switch_iteration': result['switch_iteration'] if result['switch_iteration'] is not None else '',
                'locked_class': result['locked_class'] if result['locked_class'] is not None else '',
                'final_margin': round(result['final_margin'], 6),
                'timestamp': datetime.now().isoformat(),
            }
            append_row(row, CSV_PATH)
            existing_keys.add(key)
            completed += 1

            status = 'OK' if result['success'] else 'FAIL'
            extra = ''
            if result['switch_iteration'] is not None:
                extra = (f" switch@{result['switch_iteration']}, "
                         f"locked={result['locked_class']}")
            print(f"{result['iterations']} iters | {status} | "
                  f"margin={result['final_margin']:.4f}{extra}")

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Part {args.part} complete in {elapsed:.0f}s")
    print(f"Results: {CSV_PATH}")
    print(f"Completed: {completed}/{total_runs} runs")


if __name__ == '__main__':
    main()
