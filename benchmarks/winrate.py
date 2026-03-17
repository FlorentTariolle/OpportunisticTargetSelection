"""CDF benchmark: success rate vs query budget.

Runs SimBA and SquareAttack on ResNet-50 (standard) across 100 random ImageNet
val images in three modes: untargeted, targeted-oracle, and opportunistic.
Each (method, image, mode) combo runs once at a fixed 15K budget.  The CDF is
derived in analyze_winrate.py from the recorded iteration counts.

Usage:
    python benchmark_winrate.py                   # Full run (100 images)
    python benchmark_winrate.py --n-images 2      # Smoke test
    python benchmark_winrate.py --clear           # Clear previous CSV
"""

import argparse
import csv
import random
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmarks.benchmark import load_benchmark_model, load_benchmark_image, get_true_label
from src.attacks.simba import SimBA
from src.attacks.square import SquareAttack

# ===========================================================================
# Configuration
# ===========================================================================
MODEL_NAME = 'resnet50'
SOURCE = 'standard'
EPSILON = 8 / 255
MAX_BUDGET = 15_000
STABILITY_THRESHOLD = {'SimBA': 10, 'SquareAttack': 8}
VAL_DIR = Path('data/imagenet/val')
RESULTS_DIR = Path('results')
CSV_PATH = RESULTS_DIR / 'benchmark_winrate.csv'

CSV_COLUMNS = [
    'method', 'image', 'true_label', 'mode', 'budget',
    'iterations', 'success', 'adversarial_class', 'oracle_target',
    'switch_iteration', 'locked_class', 'timestamp',
]


# ===========================================================================
# Image selection
# ===========================================================================
def select_images(val_dir: Path, n: int, seed: int) -> list[Path]:
    """Select n random images from ImageNet val directory.

    Globs for *.JPEG and *.jpeg files, samples with the given seed,
    and returns a sorted list for deterministic order.
    """
    all_images = sorted(
        list(val_dir.glob('**/*.JPEG')) + list(val_dir.glob('**/*.jpeg'))
    )
    # Deduplicate (case-insensitive filesystems might double-count)
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
    """Load existing (method, image, mode, budget) keys for resume."""
    keys = set()
    if not path.exists():
        return keys
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            keys.add((row['method'], row['image'], row['mode'], row['budget']))
    return keys


def import_opportunistic_from_ablation(
    ablation_csv: Path, winrate_csv: Path, existing_keys: set,
) -> int:
    """Import opportunistic rows from ablation CSV where S matches winrate config.

    Maps ablation rows (SimBA S=10, SquareAttack S=8) to winrate format.
    Returns number of rows imported.
    """
    if not ablation_csv.exists():
        return 0
    s_map = STABILITY_THRESHOLD  # {'SimBA': 10, 'SquareAttack': 8}
    imported = 0
    with open(ablation_csv, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row['method']
            s_val = int(row['s_value'])
            if s_val != s_map.get(method):
                continue
            image = row['image']
            budget = str(MAX_BUDGET)
            key = (method, image, 'opportunistic', budget)
            if key in existing_keys:
                continue
            winrate_row = {
                'method': method,
                'image': image,
                'true_label': row['true_label'],
                'mode': 'opportunistic',
                'budget': budget,
                'iterations': row['iterations'],
                'success': row['success'],
                'adversarial_class': row['adversarial_class'],
                'oracle_target': row.get('locked_class', ''),
                'switch_iteration': row['switch_iteration'],
                'locked_class': row['locked_class'],
                'timestamp': row['timestamp'],
            }
            append_row(winrate_row, winrate_csv)
            existing_keys.add(key)
            imported += 1
    return imported


def lookup_oracle_targets(path: Path) -> dict:
    """Load oracle targets from existing CSV rows.

    Returns dict mapping (method, image) -> oracle_target (int).
    Looks at untargeted rows for SimBA and oracle_probe rows for SquareAttack.
    """
    targets = {}
    if not path.exists():
        return targets
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row['method']
            image = row['image']
            # SimBA: oracle from successful untargeted run or oracle_target column
            if method == 'SimBA' and ('SimBA', image) not in targets:
                if row['mode'] == 'untargeted':
                    success = str(row.get('success', '')).lower() == 'true'
                    if success:
                        adv_cls = row.get('adversarial_class', '')
                        if adv_cls and adv_cls != '':
                            targets[('SimBA', image)] = int(float(adv_cls))
                if row['mode'] in ('targeted', 'opportunistic'):
                    ot = row.get('oracle_target', '')
                    if ot and ot != '':
                        targets[('SimBA', image)] = int(float(ot))
            # SquareAttack: oracle target stored in oracle_target column
            if method == 'SquareAttack' and row['mode'] == 'oracle_probe':
                ot = row.get('oracle_target', '')
                if ot and ot != '':
                    targets[('SquareAttack', image)] = int(float(ot))
    return targets


# ===========================================================================
# Attack helpers
# ===========================================================================
def run_attack(model, method, x, y_true_tensor, mode, target_class, budget,
               device):
    """Run a single attack and return a result dict.

    Args:
        model: The model (accepts [0,1] input).
        method: 'SimBA' or 'SquareAttack'.
        x: Input tensor (1, 3, 224, 224) in [0,1].
        y_true_tensor: True label tensor (1,).
        mode: 'untargeted', 'targeted', 'opportunistic', or 'oracle_probe'.
        target_class: int or None (required for targeted).
        budget: Max iterations for this run.
        device: torch device.

    Returns:
        dict with keys: iterations, success, adversarial_class,
        switch_iteration, locked_class.
    """
    is_targeted = (mode == 'targeted')
    is_opportunistic = (mode == 'opportunistic')
    target_tensor = None
    if is_targeted and target_class is not None:
        target_tensor = torch.tensor([target_class], device=device)

    y_true_int = y_true_tensor.item()

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
        targeted=is_targeted,
        target_class=target_tensor,
        early_stop=True,
        opportunistic=is_opportunistic,
        stability_threshold=STABILITY_THRESHOLD[method],
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

    if is_targeted:
        success = (pred == target_class)
    else:
        success = (pred != y_true_int)

    # Switch iteration and locked class (opportunistic only)
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


def determine_oracle_target(model, method, x, y_true_tensor, budget, device):
    """Run untargeted attack to determine oracle target class.

    Returns adversarial class if attack succeeds, else argmax non-true logit.
    """
    result = run_attack(model, method, x, y_true_tensor, 'untargeted',
                        None, budget, device)
    y_true_int = y_true_tensor.item()
    if result['success']:
        return result['adversarial_class']
    # Fallback: most-predicted non-true class
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        probs[0][y_true_int] = -1.0
        return probs.argmax(dim=1).item()


# ===========================================================================
# Row construction
# ===========================================================================
def make_row(method, image_name, true_label, mode, budget, result,
             oracle_target=None):
    """Build a CSV row dict from attack result."""
    return {
        'method': method,
        'image': image_name,
        'true_label': true_label,
        'mode': mode,
        'budget': budget,
        'iterations': result['iterations'],
        'success': result['success'],
        'adversarial_class': result['adversarial_class'],
        'oracle_target': oracle_target if oracle_target is not None else '',
        'switch_iteration': result['switch_iteration'] if result['switch_iteration'] is not None else '',
        'locked_class': result['locked_class'] if result['locked_class'] is not None else '',
        'timestamp': datetime.now().isoformat(),
    }


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="CDF benchmark: success rate vs query budget"
    )
    parser.add_argument('--clear', action='store_true',
                        help="Delete previous CSV results before running")
    parser.add_argument('--n-images', type=int, default=100,
                        help="Number of images to use (default: 100)")
    parser.add_argument('--image-seed', type=int, default=42,
                        help="Seed for image selection (default: 42)")
    parser.add_argument('--method', choices=['SimBA', 'SquareAttack'],
                        default=None,
                        help="Run only one method (default: both)")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Device: {device}")
    print(f"Model: {MODEL_NAME} ({SOURCE})")
    print(f"Epsilon: {EPSILON:.6f} ({EPSILON * 255:.0f}/255)")
    print(f"Images: {args.n_images} (seed={args.image_seed})")
    print(f"Budget: {MAX_BUDGET} (both methods)")
    print(f"Stability threshold: {STABILITY_THRESHOLD}")

    # Total runs: 3*N (SimBA) + N (oracle) + 3*N (SquareAttack) = 7N
    total_runs = args.n_images * 7
    print(f"Total runs (worst case): {total_runs}")
    print()

    RESULTS_DIR.mkdir(exist_ok=True)
    csv_path = CSV_PATH

    if args.clear and csv_path.exists():
        csv_path.unlink()
        print("Cleared previous results")

    existing_keys = load_existing_keys(csv_path)

    # Import opportunistic rows from ablation benchmark
    ablation_csv = RESULTS_DIR / 'benchmark_ablation_s.csv'
    imported = import_opportunistic_from_ablation(
        ablation_csv, csv_path, existing_keys)
    if imported:
        print(f"Imported {imported} opportunistic rows from ablation benchmark")

    oracle_targets = lookup_oracle_targets(csv_path)
    if existing_keys:
        print(f"Resuming: found {len(existing_keys)} existing results")

    # ------------------------------------------------------------------
    # Load model and images
    # ------------------------------------------------------------------
    print(f"\nLoading model: {MODEL_NAME} ({SOURCE})...")
    model = load_benchmark_model(MODEL_NAME, SOURCE, device)

    print(f"Selecting {args.n_images} images from {VAL_DIR}...")
    image_paths = select_images(VAL_DIR, args.n_images, args.image_seed)

    # Preload all images and true labels
    images = []
    for path in image_paths:
        x = load_benchmark_image(path, device)
        y_true = get_true_label(model, x)
        image_name = path.name
        images.append((image_name, x, y_true))
        print(f"  {image_name}: true_label={y_true}")

    n_images = len(images)
    completed = len(existing_keys)
    start_time = time.time()

    run_simba = args.method in (None, 'SimBA')
    run_square = args.method in (None, 'SquareAttack')

    # ------------------------------------------------------------------
    # Phase 1: SimBA (single run per mode per image)
    # ------------------------------------------------------------------
    if run_simba:
        print(f"\n{'='*70}")
        print("Phase 1: SimBA (3 modes x {} images)".format(n_images))
        print(f"{'='*70}")

        for img_idx, (image_name, x, y_true) in enumerate(images):
            y_true_tensor = torch.tensor([y_true], device=device)

            # 1a. Untargeted
            key = ('SimBA', image_name, 'untargeted', str(MAX_BUDGET))
            if key not in existing_keys:
                result = run_attack(model, 'SimBA', x, y_true_tensor,
                                    'untargeted', None, MAX_BUDGET, device)
                row = make_row('SimBA', image_name, y_true, 'untargeted',
                               MAX_BUDGET, result)
                append_row(row, csv_path)
                completed += 1
                status = 'OK' if result['success'] else 'FAIL'
                print(f"[{completed}/{total_runs}] SimBA untargeted | "
                      f"{image_name} | {result['iterations']} iters | {status}")
                # Cache oracle target
                if result['success']:
                    oracle_targets[('SimBA', image_name)] = result['adversarial_class']
            else:
                # Recover oracle target from cache
                pass

            # Determine SimBA oracle target (cached from CSV or untargeted run above)
            simba_oracle = oracle_targets.get(('SimBA', image_name))
            if simba_oracle is None:
                print(f"  [oracle] SimBA {image_name}: running untargeted probe...")
                simba_oracle = determine_oracle_target(
                    model, 'SimBA', x, y_true_tensor, MAX_BUDGET, device)
                oracle_targets[('SimBA', image_name)] = simba_oracle

            # 1b. Targeted
            key = ('SimBA', image_name, 'targeted', str(MAX_BUDGET))
            if key not in existing_keys:
                result = run_attack(model, 'SimBA', x, y_true_tensor,
                                    'targeted', simba_oracle, MAX_BUDGET, device)
                row = make_row('SimBA', image_name, y_true, 'targeted',
                               MAX_BUDGET, result, oracle_target=simba_oracle)
                append_row(row, csv_path)
                completed += 1
                status = 'OK' if result['success'] else 'FAIL'
                print(f"[{completed}/{total_runs}] SimBA targeted | "
                      f"{image_name} | {result['iterations']} iters | {status} "
                      f"(target={simba_oracle})")

            # 1c. Opportunistic
            key = ('SimBA', image_name, 'opportunistic', str(MAX_BUDGET))
            if key not in existing_keys:
                result = run_attack(model, 'SimBA', x, y_true_tensor,
                                    'opportunistic', None, MAX_BUDGET, device)
                row = make_row('SimBA', image_name, y_true, 'opportunistic',
                               MAX_BUDGET, result)
                append_row(row, csv_path)
                completed += 1
                status = 'OK' if result['success'] else 'FAIL'
                extra = ''
                if result['switch_iteration'] is not None:
                    extra = (f" (switch@{result['switch_iteration']}, "
                             f"locked={result['locked_class']})")
                print(f"[{completed}/{total_runs}] SimBA opportunistic | "
                      f"{image_name} | {result['iterations']} iters | "
                      f"{status}{extra}")

    if run_square:
        # ------------------------------------------------------------------
        # Phase 2: SquareAttack oracle probes
        # ------------------------------------------------------------------
        print(f"\n{'='*70}")
        print("Phase 2: SquareAttack oracle probes ({} images)".format(n_images))
        print(f"{'='*70}")

        for img_idx, (image_name, x, y_true) in enumerate(images):
            y_true_tensor = torch.tensor([y_true], device=device)

            key = ('SquareAttack', image_name, 'oracle_probe', str(MAX_BUDGET))
            if key not in existing_keys:
                result = run_attack(model, 'SquareAttack', x, y_true_tensor,
                                    'untargeted', None, MAX_BUDGET, device)
                # Determine oracle target
                if result['success']:
                    sq_oracle = result['adversarial_class']
                else:
                    sq_oracle = determine_oracle_target(
                        model, 'SquareAttack', x, y_true_tensor,
                        MAX_BUDGET, device)
                oracle_targets[('SquareAttack', image_name)] = sq_oracle

                row = make_row('SquareAttack', image_name, y_true, 'oracle_probe',
                               MAX_BUDGET, result, oracle_target=sq_oracle)
                append_row(row, csv_path)
                completed += 1
                status = 'OK' if result['success'] else 'FAIL'
                print(f"[{completed}/{total_runs}] SqAtk oracle_probe | "
                      f"{image_name} | {result['iterations']} iters | {status} "
                      f"(oracle={sq_oracle})")
            else:
                # Recover oracle target from cache
                if ('SquareAttack', image_name) not in oracle_targets:
                    sq_oracle = determine_oracle_target(
                        model, 'SquareAttack', x, y_true_tensor,
                        MAX_BUDGET, device)
                    oracle_targets[('SquareAttack', image_name)] = sq_oracle

        # ------------------------------------------------------------------
        # Phase 3: SquareAttack (single run per mode per image)
        # ------------------------------------------------------------------
        print(f"\n{'='*70}")
        print("Phase 3: SquareAttack (3 modes x {} images)".format(n_images))
        print(f"{'='*70}")

        for img_idx, (image_name, x, y_true) in enumerate(images):
            y_true_tensor = torch.tensor([y_true], device=device)
            sq_oracle = oracle_targets.get(('SquareAttack', image_name))

            # 3a. Untargeted
            key = ('SquareAttack', image_name, 'untargeted', str(MAX_BUDGET))
            if key not in existing_keys:
                result = run_attack(model, 'SquareAttack', x, y_true_tensor,
                                    'untargeted', None, MAX_BUDGET, device)
                row = make_row('SquareAttack', image_name, y_true, 'untargeted',
                               MAX_BUDGET, result)
                append_row(row, csv_path)
                completed += 1
                status = 'OK' if result['success'] else 'FAIL'
                print(f"[{completed}/{total_runs}] SqAtk untargeted | "
                      f"{image_name} | {result['iterations']} iters | {status}")

            # 3b. Targeted
            key = ('SquareAttack', image_name, 'targeted', str(MAX_BUDGET))
            if key not in existing_keys:
                target = sq_oracle
                result = run_attack(model, 'SquareAttack', x, y_true_tensor,
                                    'targeted', target, MAX_BUDGET, device)
                row = make_row('SquareAttack', image_name, y_true, 'targeted',
                               MAX_BUDGET, result, oracle_target=sq_oracle)
                append_row(row, csv_path)
                completed += 1
                status = 'OK' if result['success'] else 'FAIL'
                print(f"[{completed}/{total_runs}] SqAtk targeted | "
                      f"{image_name} | {result['iterations']} iters | {status} "
                      f"(target={sq_oracle})")

            # 3c. Opportunistic
            key = ('SquareAttack', image_name, 'opportunistic', str(MAX_BUDGET))
            if key not in existing_keys:
                result = run_attack(model, 'SquareAttack', x, y_true_tensor,
                                    'opportunistic', None, MAX_BUDGET, device)
                row = make_row('SquareAttack', image_name, y_true, 'opportunistic',
                               MAX_BUDGET, result)
                append_row(row, csv_path)
                completed += 1
                status = 'OK' if result['success'] else 'FAIL'
                extra = ''
                if result['switch_iteration'] is not None:
                    extra = (f" (switch@{result['switch_iteration']}, "
                             f"locked={result['locked_class']})")
                print(f"[{completed}/{total_runs}] SqAtk opportunistic | "
                      f"{image_name} | {result['iterations']} iters | "
                      f"{status}{extra}")

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Benchmark complete in {elapsed:.0f}s")
    print(f"Results: {csv_path}")
    print(f"Completed: {completed} runs")


if __name__ == '__main__':
    main()
