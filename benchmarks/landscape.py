"""Landscape analysis benchmark (Issue #18).

Re-runs SquareAttack (CE loss) on a subset of images for both standard
ResNet-50 and robust Salman2020Do_R50, saving full per-iteration
confidence_history to JSON for landscape analysis.

Usage:
    python benchmark_landscape.py                  # Full run (20 images)
    python benchmark_landscape.py --n-images 2     # Smoke test
    python benchmark_landscape.py --clear          # Clear previous results
"""

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmarks.benchmark import load_benchmark_model, load_benchmark_image, get_true_label
from src.attacks.square import SquareAttack

# ===========================================================================
# Configuration
# ===========================================================================
MODELS = {
    'standard': 'resnet50',
    'robust': 'Salman2020Do_R50',
}
EPSILON = 8 / 255
MAX_BUDGET = 10_000
STABILITY_THRESHOLD = {
    'standard': 8,
    'robust': 10,
}
VAL_DIR = Path('data/imagenet/val')
RESULTS_DIR = Path('results/landscape')


# ===========================================================================
# Image selection (same as benchmark_winrate.py for consistency)
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
            f"Found only {len(all_images)} images in {val_dir}, need {n}."
        )
    rng = random.Random(seed)
    selected = rng.sample(all_images, n)
    return sorted(selected)


# ===========================================================================
# Attack runner
# ===========================================================================
def run_and_save(model, source, x, y_true_tensor, image_name, mode, device):
    """Run SquareAttack and save confidence_history to JSON."""
    is_opportunistic = (mode == 'opportunistic')
    s_thresh = STABILITY_THRESHOLD[source]

    attack = SquareAttack(
        model=model, epsilon=EPSILON, max_iterations=MAX_BUDGET,
        device=device, loss='ce', normalize=False, seed=0,
    )

    x_adv = attack.generate(
        x, y_true_tensor,
        track_confidence=True,
        targeted=False,
        early_stop=True,
        opportunistic=is_opportunistic,
        stability_threshold=s_thresh,
    )

    # Extract results
    conf_hist = attack.confidence_history
    if conf_hist and conf_hist.get('iterations'):
        iterations = conf_hist['iterations'][-1]
    else:
        iterations = MAX_BUDGET

    with torch.no_grad():
        logits = model(x_adv)
        pred = logits.argmax(dim=1).item()
    success = (pred != y_true_tensor.item())

    # Convert confidence_history to JSON-serializable format
    # (top_classes has int keys from torch tensor indices)
    json_data = {
        'source': source,
        'model': MODELS[source],
        'image': image_name,
        'mode': mode,
        'budget': MAX_BUDGET,
        'stability_threshold': s_thresh,
        'iterations': iterations,
        'success': success,
        'adversarial_class': pred,
        'true_label': y_true_tensor.item(),
        'switch_iteration': conf_hist.get('switch_iteration'),
        'locked_class': conf_hist.get('locked_class'),
        'confidence_history': {
            'iterations': conf_hist['iterations'],
            'original_class': conf_hist['original_class'],
            'max_other_class': conf_hist['max_other_class'],
            'max_other_class_id': conf_hist['max_other_class_id'],
            'top_classes': [
                {str(k): v for k, v in d.items()}
                for d in conf_hist.get('top_classes', [])
            ],
        },
        'timestamp': datetime.now().isoformat(),
    }

    # Save to JSON
    out_path = RESULTS_DIR / f"{source}_{image_name}_{mode}.json"
    with open(out_path, 'w') as f:
        json.dump(json_data, f, indent=2)

    return iterations, success


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Landscape analysis benchmark (standard vs robust)"
    )
    parser.add_argument('--part', type=int, required=True, choices=[1, 2],
                        help="Part 1 = first half, Part 2 = second half")
    parser.add_argument('--n-images', type=int, default=20)
    parser.add_argument('--image-seed', type=int, default=42)
    parser.add_argument('--clear', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.clear:
        for f in RESULTS_DIR.glob('*.json'):
            f.unlink()
        print("Cleared previous results")

    # Select and split images
    all_images = select_images(VAL_DIR, args.n_images, args.image_seed)
    half = args.n_images // 2
    if args.part == 1:
        image_paths = all_images[:half]
    else:
        image_paths = all_images[half:]

    n_images = len(image_paths)

    print(f"Device: {device}")
    print(f"Images: {n_images} (part {args.part} of {args.n_images}, seed={args.image_seed})")
    print(f"Budget: {MAX_BUDGET}")
    print(f"Models: {MODELS}")
    print()

    total_runs = n_images * 2 * 2  # 2 sources × 2 modes
    completed = 0
    start_time = time.time()

    for source in ['standard', 'robust']:
        model_name = MODELS[source]
        print(f"\n{'='*70}")
        print(f"Loading model: {model_name} ({source})")
        print(f"{'='*70}")
        model = load_benchmark_model(model_name, source, device)

        for path in image_paths:
            image_name = path.name
            x = load_benchmark_image(path, device)
            y_true = get_true_label(model, x)
            y_true_tensor = torch.tensor([y_true], device=device)

            for mode in ['untargeted', 'opportunistic']:
                out_path = RESULTS_DIR / f"{source}_{image_name}_{mode}.json"
                if out_path.exists():
                    completed += 1
                    continue

                iters, success = run_and_save(
                    model, source, x, y_true_tensor, image_name, mode, device
                )
                completed += 1
                status = 'OK' if success else 'FAIL'
                print(f"[{completed}/{total_runs}] {source} {mode:>14s} | "
                      f"{image_name} | {iters} iters | {status}")

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.0f}s — {completed} runs")
    print(f"Results: {RESULTS_DIR}/")


if __name__ == '__main__':
    main()
