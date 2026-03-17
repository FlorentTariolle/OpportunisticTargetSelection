"""Benchmark script for opportunistic adversarial attacks.

Runs SimBA and SquareAttack (CE loss) across multiple models on ImageNet
validation images in three modes: untargeted, targeted-oracle, and opportunistic.

Split by image ranges for parallel execution:
  --image-start 0 --image-end 10   runs images 0-9

Usage:
    python benchmark.py --image-start 0 --image-end 50 --source standard
    python benchmark.py --image-start 50 --image-end 100 --source standard
    python benchmark.py --image-start 0 --image-end 4   # smoke test
    python benchmark.py --clear --image-start 0 --image-end 50 --source robust
"""

import argparse
import csv
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from torchvision import transforms

from src.models.loader import load_pretrained_model, load_robustbench_model, NormalizedModel
from src.attacks.simba import SimBA
from src.attacks.square import SquareAttack
from src.attacks.bandits import BanditsAttack
from src.utils.imaging import IMAGENET_MEAN, IMAGENET_STD

# ===========================================================================
# Configuration
# ===========================================================================
STANDARD_MODELS = ['resnet18', 'resnet50', 'vgg16', 'alexnet', 'vit_b_16']
ROBUST_MODELS = ['Salman2020Do_R18', 'Salman2020Do_R50']

EPSILONS = [8 / 255]
SEEDS = [0]
MAX_ITERATIONS = 10_000
STABILITY_THRESHOLD = {
    'standard': {'SimBA': 10, 'SquareAttack': 8, 'Bandits': 15},
    'robust': {'SimBA': 10, 'SquareAttack': 10, 'Bandits': 15},
}
VAL_DIR = Path('data/imagenet/val')
RESULTS_DIR = Path('results')

CSV_COLUMNS = [
    'model', 'method', 'epsilon', 'seed', 'image', 'mode',
    'iterations', 'success', 'adversarial_class', 'oracle_target',
    'switch_iteration', 'locked_class', 'true_conf_final', 'adv_conf_final',
    # Progress metrics
    'true_conf_initial', 'max_other_conf_initial', 'max_other_conf_final',
    'confusion_initial', 'confusion_final', 'confusion_gain',
    'peak_adv_conf', 'peak_adv_class', 'peak_adv_iter',
    'stability_threshold',
    'timestamp',
]


# ===========================================================================
# Model loading
# ===========================================================================
def load_benchmark_model(name: str, source: str, device: torch.device):
    """Load a model that accepts [0,1] input regardless of source."""
    if source == 'standard':
        raw = load_pretrained_model(name, device=device)
        model = NormalizedModel(raw, IMAGENET_MEAN, IMAGENET_STD).to(device)
    else:
        model = load_robustbench_model(name, device=device)
    model.eval()
    return model


# ===========================================================================
# Image loading
# ===========================================================================
def load_benchmark_image(path: Path, device: torch.device):
    """Load an image as a [0,1] tensor and return (x, true_label).

    x has shape (1, 3, 224, 224) in [0,1].
    true_label is an int.
    """
    img = Image.open(path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # -> [0,1]
    ])
    x = transform(img).unsqueeze(0).to(device)
    return x


def get_true_label(model, x: torch.Tensor) -> int:
    """Get model's prediction on clean image (serves as true label)."""
    with torch.no_grad():
        logits = model(x)
        return logits.argmax(dim=1).item()


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
# Attack factory
# ===========================================================================
def create_attack(method: str, model, epsilon: float, seed: int, device):
    """Create an attack instance. Both sources use [0,1] models."""
    if method == 'SimBA':
        return SimBA(
            model=model,
            epsilon=epsilon,
            max_iterations=MAX_ITERATIONS,
            device=device,
            use_dct=True,
            pixel_range=(0.0, 1.0),
        )
    elif method == 'SquareAttack':
        return SquareAttack(
            model=model,
            epsilon=epsilon,
            max_iterations=MAX_ITERATIONS,
            device=device,
            loss='ce',
            normalize=False,
            seed=seed,
        )
    elif method == 'Bandits':
        return BanditsAttack(
            model=model,
            epsilon=epsilon,
            max_iterations=5000,  # 2 queries/iter → 10K query budget
            device=device,
            pixel_range=(0.0, 1.0),
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown method: {method}")


# ===========================================================================
# Single attack run
# ===========================================================================
def run_single_attack(model, attack, x, y_true_tensor, mode, target_class,
                      seed, stability_threshold):
    """Execute a single attack and extract metrics.

    Args:
        model: The model.
        attack: Attack instance (SimBA or SquareAttack).
        x: Input tensor (1, 3, 224, 224) in [0,1].
        y_true_tensor: True label tensor (1,).
        mode: 'untargeted', 'targeted', or 'opportunistic'.
        target_class: int or None — required for targeted mode.
        seed: Random seed for reproducibility.
        stability_threshold: Consecutive stable iterations before switching.

    Returns:
        dict with attack result metrics.
    """
    # Seed for SimBA (SquareAttack seeds via constructor)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    is_targeted = (mode == 'targeted')
    is_opportunistic = (mode == 'opportunistic')
    target_tensor = None
    if is_targeted and target_class is not None:
        target_tensor = torch.tensor([target_class], device=x.device)

    y_true_int = y_true_tensor.item()

    # --- Compute initial (clean-image) metrics ---
    with torch.no_grad():
        init_logits = model(x)
        init_probs = F.softmax(init_logits, dim=1)
        true_conf_initial = init_probs[0][y_true_int].item()
        init_probs_excl = init_probs[0].clone()
        init_probs_excl[y_true_int] = -1.0
        max_other_conf_initial = init_probs_excl.max().item()
    confusion_initial = 1.0 - max(true_conf_initial - max_other_conf_initial, 0.0)

    # --- Run attack ---
    x_adv = attack.generate(
        x, y_true_tensor,
        track_confidence=True,
        targeted=is_targeted,
        target_class=target_tensor,
        early_stop=True,
        opportunistic=is_opportunistic,
        stability_threshold=stability_threshold,
    )

    # Extract iteration count
    conf_hist = attack.confidence_history
    if conf_hist and conf_hist.get('iterations'):
        iterations = conf_hist['iterations'][-1]
    else:
        iterations = attack.max_iterations

    # Check success + final metrics
    with torch.no_grad():
        logits = model(x_adv)
        pred = logits.argmax(dim=1).item()
        probs = F.softmax(logits, dim=1)
        true_conf_final = probs[0][y_true_int].item()

    if is_targeted:
        success = (pred == target_class)
    else:
        success = (pred != y_true_int)

    adv_conf_final = probs[0][pred].item()

    # max_other_conf_final
    probs_excl_final = probs[0].clone()
    probs_excl_final[y_true_int] = -1.0
    max_other_conf_final = probs_excl_final.max().item()

    # Confusion metrics
    confusion_final = 1.0 - max(true_conf_final - max_other_conf_final, 0.0)
    confusion_gain = confusion_final - confusion_initial

    # Peak adversarial confidence from confidence_history
    peak_adv_conf = peak_adv_class = peak_adv_iter = None
    if conf_hist and conf_hist.get('max_other_class') and conf_hist.get('max_other_class_id'):
        vals = conf_hist['max_other_class']
        ids = conf_hist['max_other_class_id']
        iters = conf_hist['iterations']
        if vals:
            peak_idx = max(range(len(vals)), key=lambda i: vals[i])
            peak_adv_conf = vals[peak_idx]
            peak_adv_class = ids[peak_idx]
            peak_adv_iter = iters[peak_idx]

    # Switch iteration and locked class (opportunistic only)
    switch_iter = None
    locked_class = None
    if conf_hist:
        switch_iter = conf_hist.get('switch_iteration')
        locked_class = conf_hist.get('locked_class')

    # Verify L-inf constraint
    linf = (x_adv - x).abs().max().item()
    eps = attack.epsilon
    if linf > eps + 1e-6:
        print(f"  WARNING: L-inf violation! {linf:.6f} > {eps:.6f}")

    return {
        'iterations': iterations,
        'success': success,
        'adversarial_class': pred,
        'switch_iteration': switch_iter,
        'locked_class': locked_class,
        'true_conf_final': round(true_conf_final, 6),
        'adv_conf_final': round(adv_conf_final, 6),
        'true_conf_initial': round(true_conf_initial, 6),
        'max_other_conf_initial': round(max_other_conf_initial, 6),
        'max_other_conf_final': round(max_other_conf_final, 6),
        'confusion_initial': round(confusion_initial, 6),
        'confusion_final': round(confusion_final, 6),
        'confusion_gain': round(confusion_gain, 6),
        'peak_adv_conf': round(peak_adv_conf, 6) if peak_adv_conf is not None else None,
        'peak_adv_class': peak_adv_class,
        'peak_adv_iter': peak_adv_iter,
    }


# ===========================================================================
# 3-mode pipeline
# ===========================================================================
def run_targeted_oracle_pipeline(model, method, eps, seed, x, y_true, device,
                                 completed_count, success_count, total_runs,
                                 model_name, image_name, csv_path, existing_keys,
                                 source):
    """Run untargeted → targeted-oracle → opportunistic for one config.

    Returns updated (completed_count, success_count).
    """
    y_true_tensor = torch.tensor([y_true], device=device)
    stability_threshold = STABILITY_THRESHOLD[source][method]

    for mode in ['untargeted', 'targeted', 'opportunistic']:
        # Check if already done (crash recovery)
        key = (model_name, method, f"{eps:.6f}", str(seed), image_name, mode)
        if key in existing_keys:
            continue

        attack = create_attack(method, model, eps, seed, device)

        # For targeted mode, determine oracle target
        oracle_target = None
        if mode == 'targeted':
            # Run a quick untargeted to find which class emerges
            # Use the same seed to get the same adversarial class as the
            # untargeted run that was already recorded
            probe_attack = create_attack(method, model, eps, seed, device)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            probe_adv = probe_attack.generate(
                x, y_true_tensor,
                track_confidence=False,
                targeted=False,
                early_stop=True,
            )
            with torch.no_grad():
                probe_logits = model(probe_adv)
                probe_pred = probe_logits.argmax(dim=1).item()

            if probe_pred != y_true:
                oracle_target = probe_pred
            else:
                # Untargeted failed — pick most-predicted non-true class
                probs = F.softmax(probe_logits, dim=1)
                probs[0][y_true] = -1.0
                oracle_target = probs.argmax(dim=1).item()

        result = run_single_attack(
            model, attack, x, y_true_tensor, mode, oracle_target, seed,
            stability_threshold,
        )

        row = {
            'model': model_name,
            'method': method,
            'epsilon': f"{eps:.6f}",
            'seed': seed,
            'image': image_name,
            'mode': mode,
            'iterations': result['iterations'],
            'success': result['success'],
            'adversarial_class': result['adversarial_class'],
            'oracle_target': oracle_target if oracle_target is not None else '',
            'switch_iteration': result['switch_iteration'] if result['switch_iteration'] is not None else '',
            'locked_class': result['locked_class'] if result['locked_class'] is not None else '',
            'true_conf_final': result['true_conf_final'],
            'adv_conf_final': result['adv_conf_final'],
            'true_conf_initial': result['true_conf_initial'],
            'max_other_conf_initial': result['max_other_conf_initial'],
            'max_other_conf_final': result['max_other_conf_final'],
            'confusion_initial': result['confusion_initial'],
            'confusion_final': result['confusion_final'],
            'confusion_gain': result['confusion_gain'],
            'peak_adv_conf': round(result['peak_adv_conf'], 6) if result['peak_adv_conf'] is not None else '',
            'peak_adv_class': result['peak_adv_class'] if result['peak_adv_class'] is not None else '',
            'peak_adv_iter': result['peak_adv_iter'] if result['peak_adv_iter'] is not None else '',
            'stability_threshold': stability_threshold,
            'timestamp': datetime.now().isoformat(),
        }

        append_result_to_csv(row, csv_path)

        completed_count += 1
        if result['success']:
            success_count += 1

        status = 'SUCCESS' if result['success'] else 'FAIL'
        iter_str = f"{result['iterations']} iters"
        conf_str = f" | conf_gain={result['confusion_gain']:.4f}"
        extra = ''
        if mode == 'opportunistic' and result['switch_iteration'] is not None:
            extra = f" (switch@{result['switch_iteration']}, locked={result['locked_class']})"
        if mode == 'targeted':
            extra = f" (target={oracle_target})"

        print(
            f"[{completed_count}/{total_runs}] {model_name} | {method} | "
            f"eps={eps:.4f} | seed={seed} | {image_name} | {mode} | "
            f"{iter_str} | {status}{conf_str}{extra}"
        )
        print(f"Successes: {success_count}/{completed_count} ({100*success_count/completed_count:.1f}%)")

    return completed_count, success_count


# ===========================================================================
# CSV I/O
# ===========================================================================
def append_result_to_csv(result: dict, path: Path):
    """Append a single result row to the CSV file."""
    file_exists = path.exists() and path.stat().st_size > 0
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


def import_ablation_robust(csv_path: Path):
    """Import R50 SquareAttack results from ablation CSV to avoid recomputation.

    Reads benchmark_ablation_s_robust.csv, converts S=10 opportunistic +
    baseline (untargeted/targeted) rows to benchmark format, and appends
    to csv_path if not already present.
    """
    ablation_path = RESULTS_DIR / 'benchmark_ablation_s_robust.csv'
    if not ablation_path.exists():
        print(f"  No ablation CSV at {ablation_path}, skipping import")
        return

    abl = pd.read_csv(ablation_path)

    # Filter: baselines (NaN s_value) + opportunistic at S=10
    baselines = abl[abl['s_value'].isna()]
    opp_s10 = abl[(abl['mode'] == 'opportunistic') & (abl['s_value'] == 10.0)]
    subset = pd.concat([baselines, opp_s10], ignore_index=True)

    if subset.empty:
        print("  No ablation rows to import")
        return

    # Load existing keys to avoid duplicates
    existing = load_existing_results(csv_path)

    eps_str = f"{EPSILONS[0]:.6f}"
    imported = 0

    for _, row in subset.iterrows():
        mode = row['mode']
        key = ('Salman2020Do_R50', 'SquareAttack', eps_str, '0',
               row['image'], mode)
        if key in existing:
            continue

        s_thresh = 10 if mode == 'opportunistic' else ''
        margin = row.get('final_margin', '')
        confusion_final = round(1.0 - float(margin), 6) if margin != '' and pd.notna(margin) else ''

        result = {
            'model': 'Salman2020Do_R50',
            'method': 'SquareAttack',
            'epsilon': eps_str,
            'seed': 0,
            'image': row['image'],
            'mode': mode,
            'iterations': row['iterations'],
            'success': row['success'],
            'adversarial_class': row.get('adversarial_class', ''),
            'oracle_target': row.get('oracle_target', ''),
            'switch_iteration': row.get('switch_iteration', ''),
            'locked_class': row.get('locked_class', ''),
            'true_conf_final': '',
            'adv_conf_final': '',
            'true_conf_initial': '',
            'max_other_conf_initial': '',
            'max_other_conf_final': '',
            'confusion_initial': '',
            'confusion_final': confusion_final,
            'confusion_gain': '',
            'peak_adv_conf': '',
            'peak_adv_class': '',
            'peak_adv_iter': '',
            'stability_threshold': s_thresh,
            'timestamp': row.get('timestamp', ''),
        }
        append_result_to_csv(result, csv_path)
        existing.add(key)
        imported += 1

    if imported:
        print(f"  Imported {imported} R50/SquareAttack rows from ablation CSV")


def load_existing_results(path: Path) -> set:
    """Load existing result keys for crash recovery.

    Returns a set of (model, method, epsilon, seed, image, mode) tuples.
    """
    keys = set()
    if not path.exists():
        return keys
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row['model'], row['method'], row['epsilon'],
                   row['seed'], row['image'], row['mode'])
            keys.add(key)
    return keys


# ===========================================================================
# Summary statistics
# ===========================================================================
def compute_summary_statistics(csv_path: Path):
    """Compute and save summary statistics from results CSV."""
    df = pd.read_csv(csv_path)
    df['iterations'] = pd.to_numeric(df['iterations'])
    df['success'] = df['success'].astype(bool)

    agg_dict = {
        'mean_iterations': ('iterations', 'mean'),
        'median_iterations': ('iterations', 'median'),
        'std_iterations': ('iterations', 'std'),
        'success_rate': ('success', 'mean'),
        'n_runs': ('success', 'count'),
    }

    # Add progress metric aggregations if columns exist
    if 'confusion_gain' in df.columns:
        df['confusion_gain'] = pd.to_numeric(df['confusion_gain'], errors='coerce')
        agg_dict['mean_confusion_gain'] = ('confusion_gain', 'mean')
        agg_dict['median_confusion_gain'] = ('confusion_gain', 'median')

    if 'peak_adv_conf' in df.columns:
        df['peak_adv_conf'] = pd.to_numeric(df['peak_adv_conf'], errors='coerce')
        agg_dict['mean_peak_adv_conf'] = ('peak_adv_conf', 'mean')

    if 'true_conf_initial' in df.columns and 'true_conf_final' in df.columns:
        df['true_conf_initial'] = pd.to_numeric(df['true_conf_initial'], errors='coerce')
        df['true_conf_final'] = pd.to_numeric(df['true_conf_final'], errors='coerce')
        df['confidence_drop'] = df['true_conf_initial'] - df['true_conf_final']
        agg_dict['mean_confidence_drop'] = ('confidence_drop', 'mean')

    summary = df.groupby(['model', 'method', 'epsilon', 'mode']).agg(
        **agg_dict
    ).reset_index()

    # Add switch rate for opportunistic mode
    opp = df[df['mode'] == 'opportunistic'].copy()
    if not opp.empty:
        opp['switched'] = opp['switch_iteration'].notna() & (opp['switch_iteration'] != '')
        switch_rate = opp.groupby(['model', 'method', 'epsilon']).agg(
            switch_rate=('switched', 'mean'),
        ).reset_index()
        switch_rate['mode'] = 'opportunistic'
        summary = summary.merge(switch_rate, on=['model', 'method', 'epsilon', 'mode'], how='left')
    else:
        summary['switch_rate'] = float('nan')

    summary_path = csv_path.with_name(csv_path.stem + '_summary.csv')
    summary.to_csv(summary_path, index=False)
    print(f"\nSummary saved to {summary_path}")
    print(summary.to_string(index=False))
    return summary


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Run adversarial attack benchmark")
    parser.add_argument('--image-start', type=int, required=True,
                        help="Start index of image range (inclusive)")
    parser.add_argument('--image-end', type=int, required=True,
                        help="End index of image range (exclusive)")
    parser.add_argument('--clear', action='store_true', help="Delete previous CSV results before running")
    parser.add_argument('--source', choices=['standard', 'robust'], default='standard',
                        help="Model source: 'standard' for torchvision, 'robust' for RobustBench")
    parser.add_argument('--n-images', type=int, default=100,
                        help="Number of images to use (default: 100)")
    parser.add_argument('--image-seed', type=int, default=42,
                        help="Seed for image selection (default: 42)")
    args = parser.parse_args()

    source = args.source
    if source == 'standard':
        models = STANDARD_MODELS
    else:
        models = ROBUST_MODELS

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Select and slice images
    all_images = select_images(VAL_DIR, args.n_images, args.image_seed)
    image_paths = all_images[args.image_start:args.image_end]

    print(f"Device: {device}")
    print(f"Source: {source}")
    print(f"Models: {models}")
    print(f"Images: {len(image_paths)} (indices {args.image_start}-{args.image_end} of {args.n_images}, seed={args.image_seed})")
    print(f"Epsilons: {[f'{e:.4f}' for e in EPSILONS]}")
    print(f"Seeds: {SEEDS}")
    print(f"Stability threshold: {STABILITY_THRESHOLD[source]}")
    print()

    RESULTS_DIR.mkdir(exist_ok=True)
    csv_path = RESULTS_DIR / f'benchmark_{source}.csv'

    if args.clear and csv_path.exists():
        csv_path.unlink()
        summary_path = csv_path.with_name(csv_path.stem + '_summary.csv')
        if summary_path.exists():
            summary_path.unlink()
        print("Cleared previous results")

    # Import ablation results for robust source (avoids recomputing R50/SquareAttack)
    if source == 'robust':
        import_ablation_robust(csv_path)

    existing_keys = load_existing_results(csv_path)
    if existing_keys:
        print(f"Resuming: found {len(existing_keys)} existing results")

    methods = ['SimBA', 'SquareAttack', 'Bandits']
    total_runs = len(models) * len(image_paths) * len(methods) * len(EPSILONS) * len(SEEDS) * 3
    print(f"Total runs (this part): {total_runs}")
    print("=" * 80)

    # Count only results for this part's images
    part_image_names = {p.name for p in image_paths}
    completed_count = sum(1 for k in existing_keys if k[4] in part_image_names)
    success_count = 0
    if existing_keys and csv_path.exists():
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['image'] in part_image_names and row['success'].lower() == 'true':
                    success_count += 1

    start_time = time.time()

    for model_name in models:
        print(f"\nLoading model: {model_name} ({source})...")
        model = load_benchmark_model(model_name, source, device)

        for image_path in image_paths:
            image_name = image_path.name
            if not image_path.exists():
                print(f"  WARNING: {image_path} not found, skipping")
                skipped = len(methods) * len(EPSILONS) * len(SEEDS) * 3
                completed_count += skipped
                continue

            x = load_benchmark_image(image_path, device)
            y_true = get_true_label(model, x)
            print(f"\n  Image: {image_name} (true class: {y_true})")

            for method in methods:
                for eps in EPSILONS:
                    for seed in SEEDS:
                        completed_count, success_count = run_targeted_oracle_pipeline(
                            model, method, eps, seed, x, y_true, device,
                            completed_count, success_count, total_runs,
                            model_name, image_name, csv_path, existing_keys,
                            source,
                        )

    elapsed = time.time() - start_time
    print(f"\n{'=' * 80}")
    print(f"Images {args.image_start}-{args.image_end} complete in {elapsed:.0f}s")
    print(f"Results: {csv_path}")

    # Generate summary
    if csv_path.exists():
        compute_summary_statistics(csv_path)


if __name__ == '__main__':
    main()
