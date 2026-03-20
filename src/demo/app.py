"""Backend logic for the adversarial attack demonstrator."""

import io
import os
import sys
from typing import Optional, Tuple, Dict
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.loader import ROBUSTBENCH_MODELS, NormalizedModel, load_pretrained_model, load_robustbench_model
from src.utils.imaging import (
    preprocess_image,
    denormalize_image,
    get_imagenet_label,
    get_imagenet_labels,
    IMAGENET_MEAN,
    IMAGENET_STD
)
from src.attacks import SimBA, SquareAttack, BanditsAttack

# Standard torchvision model choices
STANDARD_MODELS = ["resnet18", "resnet34", "resnet50", "vgg16", "vgg19", "alexnet", "vit_b_16"]


# Global model cache
_model_cache = {}
_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_cached_model(model_name: str = 'resnet18', source: str = 'standard'):
    """Get or load a cached model.

    All returned models accept [0,1] input. Standard models are wrapped
    with NormalizedModel; robust models already have a built-in normalizer.

    Args:
        model_name: Name of the model to load.
        source: 'standard' for torchvision, 'robust' for RobustBench.

    Returns:
        Loaded model in eval mode, accepting [0,1] input.
    """
    cache_key = (model_name, source)
    if cache_key not in _model_cache:
        if source == 'standard':
            raw = load_pretrained_model(model_name, device=_device)
            _model_cache[cache_key] = NormalizedModel(raw, IMAGENET_MEAN, IMAGENET_STD).to(_device)
        else:
            _model_cache[cache_key] = load_robustbench_model(model_name, device=_device)
        _model_cache[cache_key].eval()
    return _model_cache[cache_key]


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a [0,1] tensor to PIL Image.

    Args:
        tensor: Input tensor of shape (C, H, W) or (1, C, H, W) in [0,1].

    Returns:
        PIL Image with values in [0, 255] range.
    """
    # Remove batch dimension if present
    if tensor.dim() == 4:
        tensor = tensor[0]

    # Convert to numpy and transpose if needed
    if tensor.shape[0] == 3:  # CHW format
        tensor = tensor.permute(1, 2, 0)

    # Clamp to [0, 1] and convert to uint8
    tensor = torch.clamp(tensor, 0.0, 1.0)
    numpy_array = (tensor.cpu().detach().numpy() * 255).astype(np.uint8)

    return Image.fromarray(numpy_array)


def compute_perturbation_visualization(
    original: torch.Tensor,
    adversarial: torch.Tensor,
) -> Image.Image:
    """Compute and visualize perturbation (scaled for visibility).

    Args:
        original: Original image tensor in [0,1].
        adversarial: Adversarial image tensor in [0,1].

    Returns:
        PIL Image showing the perturbation scaled for visibility.
    """
    # Compute perturbation
    perturbation = adversarial - original

    # Remove batch dimension if present
    if perturbation.dim() == 4:
        perturbation = perturbation[0]

    # Convert to numpy and transpose if needed
    if perturbation.shape[0] == 3:  # CHW format
        perturbation = perturbation.permute(1, 2, 0)

    perturbation_np = perturbation.cpu().detach().numpy()

    # Scale perturbation for visibility: shift to [0, 1] range
    # Perturbations are typically in [-epsilon, epsilon], so we shift and scale
    perturbation_min = perturbation_np.min()
    perturbation_max = perturbation_np.max()

    if perturbation_max > perturbation_min:
        # Scale to [0, 1]
        perturbation_scaled = (perturbation_np - perturbation_min) / (perturbation_max - perturbation_min)
    else:
        perturbation_scaled = np.zeros_like(perturbation_np)

    # Convert to uint8
    perturbation_uint8 = (perturbation_scaled * 255).astype(np.uint8)

    return Image.fromarray(perturbation_uint8)


def create_confidence_graph(
    confidence_history: Optional[Dict],
    targeted: bool = False,
    opportunistic: bool = False
) -> Optional[Image.Image]:
    """Create a graph showing confidence evolution.

    Args:
        confidence_history: Dictionary with 'iterations', 'original_class', 'max_other_class',
                           and optionally 'target_class' keys.
        targeted: If True, show target class confidence line with "Target Class" label.
        opportunistic: If True, show locked class confidence line with "Locked Class" label.
                      (The locked class is the one chosen by the stability criterion.)

    Returns:
        PIL Image of the graph, or None if no history available.
    """
    if confidence_history is None or len(confidence_history['iterations']) == 0:
        return None

    iterations = confidence_history['iterations']
    original_conf = confidence_history['original_class']
    max_other_conf = confidence_history['max_other_class']

    # Create figure (compact for bento layout)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(iterations, original_conf, 'b-', label='True class', linewidth=1.5)
    ax.plot(iterations, max_other_conf, 'r--', label='Max other', linewidth=1.5)

    # Add target/locked class line if available
    if opportunistic:
        # For opportunistic mode, reconstruct the locked class confidence from:
        # - top_classes data (before switch): look up locked_class in each dict
        # - target_class data (after switch): direct values
        locked_class = confidence_history.get('locked_class')
        top_classes = confidence_history.get('top_classes', [])
        target_conf_after = confidence_history.get('target_class', [])

        if locked_class is not None:
            # Build the full locked class confidence history
            locked_conf = []
            # Before switch: extract from top_classes
            for top_dict in top_classes:
                if locked_class in top_dict:
                    locked_conf.append(top_dict[locked_class])
                else:
                    # Class wasn't in top 10 at this point, use 0 or skip
                    locked_conf.append(None)
            # After switch: use target_class values
            locked_conf.extend(target_conf_after)

            # Filter out None values and align with iterations
            valid_indices = [i for i, c in enumerate(locked_conf) if c is not None]
            valid_conf = [locked_conf[i] for i in valid_indices]
            valid_iters = [iterations[i] for i in valid_indices if i < len(iterations)]

            if valid_conf and valid_iters:
                ax.plot(valid_iters[:len(valid_conf)], valid_conf[:len(valid_iters)],
                       'g-', label='Locked class', linewidth=1.5)

            # Vertical line at lock iteration
            switch_iter = confidence_history.get('switch_iteration')
            if switch_iter is not None:
                ax.axvline(x=switch_iter, color='purple', linestyle=':', linewidth=1,
                           label=f'Lock (T={switch_iter})')
    elif targeted and 'target_class' in confidence_history and len(confidence_history['target_class']) > 0:
        target_conf = confidence_history['target_class']
        # For targeted mode from the start, align with first N iterations
        target_iterations = iterations[:len(target_conf)]
        ax.plot(target_iterations, target_conf, 'g-', label='Target class', linewidth=1.5)

    ax.set_xlabel('Iteration', fontsize=14)
    ax.set_ylabel('Confidence', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    plt.tight_layout()

    # Convert to PIL Image using BytesIO
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)

    return img


def predict_image(
    image: Image.Image,
    model_name: str = 'resnet18',
    source: str = 'standard',
) -> Tuple[str, float, int]:
    """Get model prediction for an image.

    Args:
        image: Input PIL Image.
        model_name: Name of the model to use for prediction.
        source: 'standard' or 'robust'.

    Returns:
        Tuple of (label, confidence, class_index).
    """
    model = get_cached_model(model_name, source=source)

    # All models accept [0,1] input (standard wrapped with NormalizedModel)
    image_tensor = preprocess_image(image, normalize=False, device=_device)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(logits, dim=1).item()
        confidence = probs[0][predicted_class].item()

    label = get_imagenet_label(predicted_class)
    return label, confidence, predicted_class


def run_attack(
    image: Optional[Image.Image],
    method: str,
    epsilon: float,
    max_iterations: int,
    model_name: str = 'resnet18',
    targeted: bool = False,
    target_class: Optional[int] = None,
    opportunistic: bool = False,
    switch_iteration: int = 5,
    loss: str = 'margin',
    source: str = 'standard',
    seed: int = 42,
) -> Tuple[Image.Image, Image.Image, Image.Image, str]:
    """Run adversarial attack on the image.

    Args:
        image: Input PIL Image to attack.
        method: Attack method name (e.g., 'SimBA').
        epsilon: Maximum perturbation magnitude (L-inf norm).
        max_iterations: Maximum number of attack iterations.
        model_name: Name of the target model to attack.
        targeted: If True, perform targeted attack.
        target_class: Target class index for targeted attack.
        opportunistic: If True, start untargeted and switch to targeted at iteration T.
        switch_iteration: Iteration at which to lock onto the leading non-true class.
        loss: Loss function for Square Attack ('margin' or 'ce').
        source: 'standard' or 'robust'.

    Returns:
        Tuple of (adversarial_image, perturbation_image, confidence_graph, result_text).
    """
    if image is None:
        return None, None, None, "Please upload an image first."

    try:
        # Load model — all models accept [0,1] input
        model = get_cached_model(model_name, source=source)

        # All models accept [0,1]; no ImageNet normalization needed
        image_tensor = preprocess_image(image, normalize=False, device=_device)

        # Get original prediction
        with torch.no_grad():
            logits = model(image_tensor)
            probs = F.softmax(logits, dim=1)
            original_class = torch.argmax(logits, dim=1).item()
            original_confidence = probs[0][original_class].item()
            # Highest non-true class confidence (for confusion metrics)
            probs_excl = probs[0].clone()
            probs_excl[original_class] = -1.0
            initial_max_other_confidence = probs_excl.max().item()

        original_label = get_imagenet_label(original_class)

        # Seed for reproducibility
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # Initialize attack — all models accept [0,1], so config is uniform
        if method == "SimBA":
            attack = SimBA(
                model=model,
                epsilon=epsilon,
                max_iterations=max_iterations,
                device=_device,
                use_dct=True,
                pixel_range=(0.0, 1.0),
            )
        elif method == "Square Attack":
            attack = SquareAttack(
                model=model,
                epsilon=epsilon,
                max_iterations=max_iterations,
                device=_device,
                loss=loss,
                normalize=False,
                seed=seed,
            )
        elif method == "Bandits":
            attack = BanditsAttack(
                model=model,
                epsilon=epsilon,
                max_iterations=max_iterations,
                device=_device,
                pixel_range=(0.0, 1.0),
                seed=seed,
            )
        else:
            return None, None, None, f"Unknown attack method: {method}"

        # Run attack with confidence tracking
        y_true = torch.tensor([original_class], device=_device)
        if targeted and target_class is not None:
            target_tensor = torch.tensor([target_class], device=_device)
            x_adv = attack.generate(
                image_tensor, y_true,
                track_confidence=True,
                targeted=True,
                target_class=target_tensor
            )
        else:
            x_adv = attack.generate(
                image_tensor, y_true,
                track_confidence=True,
                opportunistic=opportunistic,
                naive_switch_iteration=switch_iteration
            )

        # Get adversarial prediction
        with torch.no_grad():
            adv_logits = model(x_adv)
            adv_probs = F.softmax(adv_logits, dim=1)
            adv_class = torch.argmax(adv_logits, dim=1).item()
            adv_confidence = adv_probs[0][adv_class].item()
            final_true_confidence = adv_probs[0][original_class].item()
            # Highest non-true class confidence after attack
            adv_probs_excl = adv_probs[0].clone()
            adv_probs_excl[original_class] = -1.0
            final_max_other_class = torch.argmax(adv_probs_excl).item()
            final_max_other_confidence = adv_probs_excl[final_max_other_class].item()

        adv_label = get_imagenet_label(adv_class)
        final_max_other_label = get_imagenet_label(final_max_other_class)

        # Confidence metrics
        loss_of_confidence = original_confidence - final_true_confidence

        # Check if attack was successful
        if targeted and target_class is not None:
            is_successful = (adv_class == target_class)
            target_label = get_imagenet_label(target_class)
        else:
            is_successful = attack.check_adversarial(x_adv, y_true).item()

        # Convert adversarial tensor to PIL
        adv_image = tensor_to_pil(x_adv)

        # Compute perturbation visualization
        perturbation_image = compute_perturbation_visualization(
            image_tensor, x_adv
        )

        # Get confidence history and check for opportunistic switch
        ch = getattr(attack, 'confidence_history', None)
        switch_iteration = ch.get('switch_iteration') if ch else None

        # Create confidence evolution graph
        # For opportunistic mode, only show locked class line if a switch occurred
        opportunistic_switched = opportunistic and switch_iteration is not None
        confidence_graph = create_confidence_graph(
            ch,
            targeted=targeted,
            opportunistic=opportunistic_switched
        )

        # Iterations used (from confidence history if available)
        iterations_used = ch['iterations'][-1] if ch and ch.get('iterations') else None
        budget_display = f"{iterations_used} iterations / {max_iterations}" if iterations_used is not None else f"? iterations / {max_iterations}"

        # Compute perturbation metrics
        total_perturbation = (x_adv - image_tensor).squeeze(0)
        linf = total_perturbation.abs().max().item()
        l2 = total_perturbation.norm(2).item()
        mean_l1 = total_perturbation.abs().mean().item()

        # Build compact result text
        status_word = "**Success**" if is_successful else "**Failure**"

        if opportunistic and switch_iteration is not None:
            mode_line = f"OT (T={switch_iteration})"
        elif opportunistic:
            mode_line = "OT (no lock)"
        elif targeted and target_class is not None:
            target_label = get_imagenet_label(target_class)
            mode_line = f"Targeted → {target_class} ({target_label})"
        else:
            mode_line = "Untargeted"

        result_text = (
            f"{status_word} — {mode_line} | {budget_display}\n\n"
            f"**Original:** {original_class} ({original_label}) — {original_confidence:.2%}\n\n"
            f"**Result:** {adv_class} ({adv_label}) — {adv_confidence:.2%}\n\n"
            f"True: {original_confidence:.2%} → {final_true_confidence:.2%} ({loss_of_confidence:+.2%})\n\n"
            f"L∞={linf:.4f} | L2={l2:.2f}"
        )

        return adv_image, perturbation_image, confidence_graph, result_text

    except Exception as e:
        import traceback
        error_msg = f"Error during attack: {str(e)}\n\n{traceback.format_exc()}"
        return None, None, None, error_msg
