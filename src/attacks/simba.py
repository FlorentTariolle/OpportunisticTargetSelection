"""SimBA (Simple Black-box Adversarial) attack implementation."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from .base import BaseAttack


class SimBA(BaseAttack):
    """Simple Black-box Adversarial (SimBA) attack.
    
    SimBA is a query-efficient black-box attack that uses random search
    in the DCT (Discrete Cosine Transform) space or pixel space.
    """
    
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.03,
        max_iterations: int = 1000,
        device: Optional[torch.device] = None,
        use_dct: bool = True,
        block_size: int = 8,
        pixel_range: Tuple[float, float] = (-3.0, 3.0),
    ):
        """Initialize SimBA attack.

        Args:
            model: The target model to attack.
            epsilon: Maximum perturbation magnitude (L∞ norm).
            max_iterations: Maximum number of iterations.
            device: Device to run the attack on.
            use_dct: If True, use DCT space for perturbations; else use pixel space.
            block_size: Block size for DCT transform (only used if use_dct=True).
            pixel_range: Valid value range for clipping. Use (-3.0, 3.0) for
                ImageNet-normalized inputs, (0.0, 1.0) for unnormalized / robust models.
        """
        super().__init__(model, epsilon, max_iterations, device)
        self.use_dct = use_dct
        self.block_size = block_size
        self.pixel_range = pixel_range
        # Cache for DCT basis vectors (computed on first use)
        self._dct_basis_cache = None
        self._dct_basis_shape = None
    
    def generate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        track_confidence: bool = False,
        targeted: bool = False,
        target_class: Optional[torch.Tensor] = None,
        early_stop: bool = True,
        opportunistic: bool = False,
        stability_threshold: int = 30,
        reference_direction: Optional[torch.Tensor] = None,
        naive_switch_iteration: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """Generate adversarial examples using SimBA.

        Args:
            x: Input images tensor of shape (batch_size, channels, height, width).
            y: True labels tensor of shape (batch_size,).
            track_confidence: If True, track confidence values during attack.
            targeted: If True, perform targeted attack towards target_class.
            target_class: Target class tensor of shape (batch_size,). Required if targeted=True.
            early_stop: If True, stop as soon as the attack succeeds. If False, run all max_iterations.
            opportunistic: If True, start untargeted and switch to targeted when max class stabilizes.
            stability_threshold: Number of consecutive accepted perturbations with same max class before switching.
            **kwargs: Additional parameters (not used currently).

        Returns:
            Adversarial examples tensor.

        Raises:
            ValueError: If input shapes are invalid or batch sizes don't match.
        """
        # Input validation
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor (batch, channels, height, width), got {x.dim()}D")
        if y.dim() != 1:
            raise ValueError(f"Expected 1D tensor for labels, got {y.dim()}D")
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"Batch size mismatch: x has {x.shape[0]} samples, y has {y.shape[0]}")
        if targeted and target_class is None:
            raise ValueError("target_class is required when targeted=True")
        if targeted and target_class is not None and target_class.shape[0] != x.shape[0]:
            raise ValueError(f"Batch size mismatch: x has {x.shape[0]} samples, target_class has {target_class.shape[0]}")
        if opportunistic and targeted:
            raise ValueError("opportunistic=True cannot be combined with targeted=True (opportunistic starts untargeted)")

        batch_size = x.shape[0]
        x_adv = x.clone().to(self.device)
        y = y.to(self.device)
        if target_class is not None:
            target_class = target_class.to(self.device)

        # Initialize confidence_history
        self.confidence_history = None

        # Process each image in the batch
        if track_confidence and batch_size == 1:
            t_class = target_class[0] if targeted else None
            result = self._attack_single_image(
                x[0], y[0], track_confidence=True, targeted=targeted, target_class=t_class,
                early_stop=early_stop, opportunistic=opportunistic, stability_threshold=stability_threshold,
                reference_direction=reference_direction, naive_switch_iteration=naive_switch_iteration,
            )
            if isinstance(result, tuple) and len(result) == 2:
                x_adv[0], self.confidence_history = result
            else:
                # Fallback if result is not a tuple (shouldn't happen)
                x_adv[0] = result if not isinstance(result, tuple) else result[0]
                self.confidence_history = None
        else:
            if track_confidence:
                # If track_confidence is True but batch_size != 1, warn user
                import warnings
                warnings.warn(f"track_confidence=True is only supported for batch_size=1. Got batch_size={batch_size}. Confidence tracking disabled.")
            for i in range(batch_size):
                t_class = target_class[i] if targeted else None
                result = self._attack_single_image(x[i], y[i], track_confidence=False, targeted=targeted, target_class=t_class, early_stop=early_stop, opportunistic=opportunistic, stability_threshold=stability_threshold, reference_direction=reference_direction, naive_switch_iteration=naive_switch_iteration)
                if isinstance(result, tuple):
                    x_adv[i] = result[0]
                else:
                    x_adv[i] = result

        return x_adv
    
    def _attack_single_image(
        self,
        x: torch.Tensor,
        y_true: torch.Tensor,
        track_confidence: bool = False,
        targeted: bool = False,
        target_class: Optional[torch.Tensor] = None,
        early_stop: bool = True,
        opportunistic: bool = False,
        stability_threshold: int = 30,
        reference_direction: Optional[torch.Tensor] = None,
        naive_switch_iteration: Optional[int] = None,
    ) -> tuple:
        """Attack a single image.

        Args:
            x: Single image tensor of shape (channels, height, width).
            y_true: True label tensor (scalar).
            track_confidence: If True, track confidence values during attack.
            targeted: If True, perform targeted attack towards target_class.
            target_class: Target class tensor (scalar). Required if targeted=True.
            early_stop: If False, run all max_iterations without returning on success.
            opportunistic: If True, start untargeted and switch to targeted when max class stabilizes.
            stability_threshold: Number of consecutive accepted perturbations with same max class before switching.

        Returns:
            If track_confidence: (adversarial image tensor, confidence_history dict)
            Else: adversarial image tensor
        """
        x_adv = x.clone()
        confidence_history = {
            'iterations': [],
            'original_class': [],
            'max_other_class': [],
            'max_other_class_id': [],
            'target_class': [],
            'cos_sim_to_ref': [],
            'cos_sim_iterations': [],
            'switch_iteration': None,  # Iteration when opportunistic mode switched to targeted
            'top_classes': [],  # List of dicts {class_id: confidence} for top 10 classes (opportunistic only)
            'locked_class': None  # Class ID that was locked (opportunistic only)
        }

        # Precompute flat reference direction for cosine similarity
        ref_flat = None
        if reference_direction is not None:
            ref_flat = reference_direction.flatten().to(self.device)

        # Opportunistic targeting state
        if opportunistic:
            stability_counter = 0
            prev_max_class = None
            switched_to_targeted = False
            switch_iteration = None
        
        # Get initial confidence
        if track_confidence:
            with torch.no_grad():
                logits = self.model(x_adv.unsqueeze(0))
                probs = torch.nn.functional.softmax(logits, dim=1)
                original_conf = probs[0][y_true].item()
                # Get max confidence excluding original class
                probs_excluding_original = probs[0].clone()
                probs_excluding_original[y_true] = -1.0  # Set to -1 to exclude
                max_other_conf = probs_excluding_original.max().item()
                max_other_class_id = probs_excluding_original.argmax().item()
                confidence_history['iterations'].append(0)
                confidence_history['original_class'].append(original_conf)
                confidence_history['max_other_class'].append(max_other_conf)
                confidence_history['max_other_class_id'].append(max_other_class_id)
                if targeted and target_class is not None:
                    confidence_history['target_class'].append(probs[0][target_class].item())
                # Track top 10 classes for opportunistic mode (before locking)
                if opportunistic:
                    top10_indices = torch.topk(probs_excluding_original, k=10).indices.tolist()
                    top10_conf = {idx: probs[0][idx].item() for idx in top10_indices}
                    confidence_history['top_classes'].append(top10_conf)
        
        # Check if already successful (misclassified for untargeted, target class for targeted)
        # Skip when early_stop=False so we run the full budget (e.g. fixed 100 untargeted iters)
        if early_stop:
            if targeted:
                if self._is_target_class(x_adv.unsqueeze(0), target_class.unsqueeze(0)):
                    if track_confidence:
                        return x_adv, confidence_history
                    return x_adv
            else:
                if self._is_misclassified(x_adv.unsqueeze(0), y_true.unsqueeze(0)):
                    if track_confidence:
                        return x_adv, confidence_history
                    return x_adv
        
        # Generate candidate indices (memory-efficient)
        if self.use_dct:
            candidate_indices = self._generate_dct_candidate_indices(x)
        else:
            candidate_indices = self._generate_pixel_candidate_indices(x)
        
        # Randomly shuffle candidate indices
        num_candidates = len(candidate_indices)
        shuffled_indices = torch.randperm(num_candidates, device=self.device)
        
        # Try each candidate perturbation
        for iteration in range(min(self.max_iterations, num_candidates)):
            idx = shuffled_indices[iteration]
            candidate_idx = candidate_indices[idx]
            
            # Generate perturbation on-the-fly
            perturbation = self._create_perturbation(x.shape, candidate_idx)
            
            # Track confidence at start of iteration (before trying perturbations)
            if track_confidence:
                should_track = (
                    iteration < 50 or  # Track every iteration for first 50
                    iteration % 10 == 0  # Then every 10 iterations
                )
                if should_track:
                    with torch.no_grad():
                        logits = self.model(x_adv.unsqueeze(0))
                        probs = torch.nn.functional.softmax(logits, dim=1)
                        original_conf = probs[0][y_true].item()
                        probs_excluding_original = probs[0].clone()
                        probs_excluding_original[y_true] = -1.0
                        max_other_conf = probs_excluding_original.max().item()
                        max_other_class_id = probs_excluding_original.argmax().item()
                        confidence_history['iterations'].append(iteration + 1)
                        confidence_history['original_class'].append(original_conf)
                        confidence_history['max_other_class'].append(max_other_conf)
                        confidence_history['max_other_class_id'].append(max_other_class_id)
                        if targeted and target_class is not None:
                            confidence_history['target_class'].append(probs[0][target_class].item())
                        if ref_flat is not None:
                            delta = (x_adv - x).flatten()
                            cos = F.cosine_similarity(delta.unsqueeze(0), ref_flat.unsqueeze(0)).item()
                            confidence_history['cos_sim_to_ref'].append(cos)
                            confidence_history['cos_sim_iterations'].append(iteration + 1)
                        # Track top 10 classes for opportunistic mode (before locking)
                        if opportunistic and not switched_to_targeted:
                            top10_indices = torch.topk(probs_excluding_original, k=10).indices.tolist()
                            top10_conf = {idx: probs[0][idx].item() for idx in top10_indices}
                            confidence_history['top_classes'].append(top10_conf)
            
            # Get current confidence once (for efficiency)
            with torch.no_grad():
                logits_current = self.model(x_adv.unsqueeze(0))
                probs_current = torch.nn.functional.softmax(logits_current, dim=1)
                current_conf = probs_current[0][y_true].item()
                current_pred = torch.argmax(logits_current, dim=1).item()
                if targeted:
                    current_target_conf = probs_current[0][target_class].item()
            
            # Try positive perturbation
            x_candidate = x_adv + perturbation
            # Clip perturbation to respect epsilon constraint and valid pixel range
            # For normalized ImageNet images, use reasonable bounds
            perturbation_clipped = self.clip_perturbation(
                x_adv, perturbation, pixel_range=self.pixel_range
            )
            x_candidate = x_adv + perturbation_clipped
            x_candidate_batch = x_candidate.unsqueeze(0)

            # Check candidate in one forward pass
            with torch.no_grad():
                logits_candidate = self.model(x_candidate_batch)
                probs_candidate = torch.nn.functional.softmax(logits_candidate, dim=1)
                candidate_conf = probs_candidate[0][y_true].item()
                candidate_pred = torch.argmax(logits_candidate, dim=1).item()
                if targeted:
                    candidate_target_conf = probs_candidate[0][target_class].item()

            # Check success condition
            if targeted:
                # Targeted: success if prediction equals target class
                pos_success = (candidate_pred == target_class.item())
            else:
                # Untargeted: success if misclassified
                pos_success = (candidate_pred != y_true.item())

            if pos_success:
                x_adv = x_candidate
                if track_confidence:
                    probs_excluding_original = probs_candidate[0].clone()
                    probs_excluding_original[y_true] = -1.0
                    max_other_conf = probs_excluding_original.max().item()
                    max_other_class_id = probs_excluding_original.argmax().item()
                    if confidence_history['iterations'] and confidence_history['iterations'][-1] == iteration + 1:
                        confidence_history['original_class'][-1] = candidate_conf
                        confidence_history['max_other_class'][-1] = max_other_conf
                        confidence_history['max_other_class_id'][-1] = max_other_class_id
                        if targeted:
                            confidence_history['target_class'][-1] = candidate_target_conf
                    else:
                        confidence_history['iterations'].append(iteration + 1)
                        confidence_history['original_class'].append(candidate_conf)
                        confidence_history['max_other_class'].append(max_other_conf)
                        confidence_history['max_other_class_id'].append(max_other_class_id)
                        if targeted:
                            confidence_history['target_class'].append(candidate_target_conf)
                # Opportunistic stability check
                if opportunistic and not switched_to_targeted:
                    probs_excluding_true = probs_candidate[0].clone()
                    probs_excluding_true[y_true] = -1.0
                    current_max_class = torch.argmax(probs_excluding_true).item()
                    if naive_switch_iteration is not None and iteration + 1 >= naive_switch_iteration:
                        targeted = True
                        target_class = torch.tensor(current_max_class, device=self.device)
                        current_target_conf = probs_candidate[0][target_class].item()
                        switched_to_targeted = True
                        switch_iteration = iteration + 1
                        confidence_history['switch_iteration'] = switch_iteration
                        confidence_history['locked_class'] = current_max_class
                    elif naive_switch_iteration is None and prev_max_class is not None and current_max_class == prev_max_class:
                        stability_counter += 1
                        if stability_counter >= stability_threshold:
                            targeted = True
                            target_class = torch.tensor(current_max_class, device=self.device)
                            current_target_conf = probs_candidate[0][target_class].item()
                            switched_to_targeted = True
                            switch_iteration = iteration + 1
                            confidence_history['switch_iteration'] = switch_iteration
                            confidence_history['locked_class'] = current_max_class
                    else:
                        stability_counter = 0
                    prev_max_class = current_max_class
                if early_stop:
                    return x_adv, confidence_history if track_confidence else x_adv
                continue

            # Check acceptance criterion
            # Targeted: accept if target class confidence INCREASES
            # Untargeted: accept if true class confidence DECREASES
            if targeted:
                pos_accept = (candidate_target_conf > current_target_conf)
            else:
                pos_accept = (candidate_conf < current_conf)

            if pos_accept:
                x_adv = x_candidate
                # Opportunistic stability check
                if opportunistic and not switched_to_targeted:
                    probs_excluding_true = probs_candidate[0].clone()
                    probs_excluding_true[y_true] = -1.0
                    current_max_class = torch.argmax(probs_excluding_true).item()
                    if naive_switch_iteration is not None and iteration + 1 >= naive_switch_iteration:
                        targeted = True
                        target_class = torch.tensor(current_max_class, device=self.device)
                        current_target_conf = probs_candidate[0][target_class].item()
                        switched_to_targeted = True
                        switch_iteration = iteration + 1
                        confidence_history['switch_iteration'] = switch_iteration
                        confidence_history['locked_class'] = current_max_class
                    elif naive_switch_iteration is None and prev_max_class is not None and current_max_class == prev_max_class:
                        stability_counter += 1
                        if stability_counter >= stability_threshold:
                            targeted = True
                            target_class = torch.tensor(current_max_class, device=self.device)
                            current_target_conf = probs_candidate[0][target_class].item()
                            switched_to_targeted = True
                            switch_iteration = iteration + 1
                            confidence_history['switch_iteration'] = switch_iteration
                            confidence_history['locked_class'] = current_max_class
                    else:
                        stability_counter = 0
                    prev_max_class = current_max_class
                continue  # Accept this perturbation and move to next iteration

            # Try negative perturbation
            negative_perturbation = -perturbation
            # Clip perturbation to respect epsilon constraint and valid pixel range
            perturbation_clipped = self.clip_perturbation(
                x_adv, negative_perturbation, pixel_range=self.pixel_range
            )
            x_candidate = x_adv + perturbation_clipped
            x_candidate_batch = x_candidate.unsqueeze(0)

            # Check candidate in one forward pass
            with torch.no_grad():
                logits_candidate = self.model(x_candidate_batch)
                probs_candidate = torch.nn.functional.softmax(logits_candidate, dim=1)
                candidate_conf = probs_candidate[0][y_true].item()
                candidate_pred = torch.argmax(logits_candidate, dim=1).item()
                if targeted:
                    candidate_target_conf = probs_candidate[0][target_class].item()

            # Check success condition
            if targeted:
                # Targeted: success if prediction equals target class
                neg_success = (candidate_pred == target_class.item())
            else:
                # Untargeted: success if misclassified
                neg_success = (candidate_pred != y_true.item())

            if neg_success:
                x_adv = x_candidate
                if track_confidence:
                    probs_excluding_original = probs_candidate[0].clone()
                    probs_excluding_original[y_true] = -1.0
                    max_other_conf = probs_excluding_original.max().item()
                    max_other_class_id = probs_excluding_original.argmax().item()
                    if confidence_history['iterations'] and confidence_history['iterations'][-1] == iteration + 1:
                        confidence_history['original_class'][-1] = candidate_conf
                        confidence_history['max_other_class'][-1] = max_other_conf
                        confidence_history['max_other_class_id'][-1] = max_other_class_id
                        if targeted:
                            confidence_history['target_class'][-1] = candidate_target_conf
                    else:
                        confidence_history['iterations'].append(iteration + 1)
                        confidence_history['original_class'].append(candidate_conf)
                        confidence_history['max_other_class'].append(max_other_conf)
                        confidence_history['max_other_class_id'].append(max_other_class_id)
                        if targeted:
                            confidence_history['target_class'].append(candidate_target_conf)
                # Opportunistic stability check
                if opportunistic and not switched_to_targeted:
                    probs_excluding_true = probs_candidate[0].clone()
                    probs_excluding_true[y_true] = -1.0
                    current_max_class = torch.argmax(probs_excluding_true).item()
                    if naive_switch_iteration is not None and iteration + 1 >= naive_switch_iteration:
                        targeted = True
                        target_class = torch.tensor(current_max_class, device=self.device)
                        current_target_conf = probs_candidate[0][target_class].item()
                        switched_to_targeted = True
                        switch_iteration = iteration + 1
                        confidence_history['switch_iteration'] = switch_iteration
                        confidence_history['locked_class'] = current_max_class
                    elif naive_switch_iteration is None and prev_max_class is not None and current_max_class == prev_max_class:
                        stability_counter += 1
                        if stability_counter >= stability_threshold:
                            targeted = True
                            target_class = torch.tensor(current_max_class, device=self.device)
                            current_target_conf = probs_candidate[0][target_class].item()
                            switched_to_targeted = True
                            switch_iteration = iteration + 1
                            confidence_history['switch_iteration'] = switch_iteration
                            confidence_history['locked_class'] = current_max_class
                    else:
                        stability_counter = 0
                    prev_max_class = current_max_class
                if early_stop:
                    return x_adv, confidence_history if track_confidence else x_adv
                continue

            # Check acceptance criterion
            # Targeted: accept if target class confidence INCREASES
            # Untargeted: accept if true class confidence DECREASES
            if targeted:
                neg_accept = (candidate_target_conf > current_target_conf)
            else:
                neg_accept = (candidate_conf < current_conf)

            if neg_accept:
                x_adv = x_candidate
                # Opportunistic stability check
                if opportunistic and not switched_to_targeted:
                    probs_excluding_true = probs_candidate[0].clone()
                    probs_excluding_true[y_true] = -1.0
                    current_max_class = torch.argmax(probs_excluding_true).item()
                    if naive_switch_iteration is not None and iteration + 1 >= naive_switch_iteration:
                        targeted = True
                        target_class = torch.tensor(current_max_class, device=self.device)
                        current_target_conf = probs_candidate[0][target_class].item()
                        switched_to_targeted = True
                        switch_iteration = iteration + 1
                        confidence_history['switch_iteration'] = switch_iteration
                        confidence_history['locked_class'] = current_max_class
                    elif naive_switch_iteration is None and prev_max_class is not None and current_max_class == prev_max_class:
                        stability_counter += 1
                        if stability_counter >= stability_threshold:
                            targeted = True
                            target_class = torch.tensor(current_max_class, device=self.device)
                            current_target_conf = probs_candidate[0][target_class].item()
                            switched_to_targeted = True
                            switch_iteration = iteration + 1
                            confidence_history['switch_iteration'] = switch_iteration
                            confidence_history['locked_class'] = current_max_class
                    else:
                        stability_counter = 0
                    prev_max_class = current_max_class
                continue  # Accept this perturbation and move to next iteration

        # Exhausted loop: record final state for benchmarking
        num_done = min(self.max_iterations, num_candidates)
        if track_confidence and (not confidence_history['iterations'] or confidence_history['iterations'][-1] != num_done):
            with torch.no_grad():
                logits = self.model(x_adv.unsqueeze(0))
                probs = torch.nn.functional.softmax(logits, dim=1)
                final_original_conf = probs[0][y_true].item()
                probs_excl = probs[0].clone()
                probs_excl[y_true] = -1.0
                final_max_other_conf = probs_excl.max().item()
                final_max_other_class_id = probs_excl.argmax().item()
            confidence_history['iterations'].append(num_done)
            confidence_history['original_class'].append(final_original_conf)
            confidence_history['max_other_class'].append(final_max_other_conf)
            confidence_history['max_other_class_id'].append(final_max_other_class_id)
        if track_confidence:
            return x_adv, confidence_history
        return x_adv
    
    def _dct_2d(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 2D DCT-II transform to input tensor.
        
        Args:
            x: Input tensor of shape (..., height, width).
        
        Returns:
            DCT coefficients tensor of same shape.
        """
        # DCT-II implementation using matrix multiplication
        # Standard 2D DCT-II formula with proper normalization
        N = x.shape[-1]  # Assuming square blocks
        device = x.device
        dtype = x.dtype
        
        # Create DCT matrix
        i = torch.arange(N, device=device, dtype=dtype).unsqueeze(0)  # (1, N)
        u = torch.arange(N, device=device, dtype=dtype).unsqueeze(1)  # (N, 1)
        
        # DCT matrix: cos(pi * u * (2i + 1) / (2*N))
        # Normalization factors: alpha(0) = 1/sqrt(2), alpha(k) = 1 for k>0
        # Combined with (2/N) factor: sqrt(2/N) for u=0, sqrt(2/N) for u>0
        # But standard DCT-II uses: sqrt(1/N) for u=0, sqrt(2/N) for u>0
        dct_matrix = torch.cos(math.pi * u * (2 * i + 1) / (2 * N))
        dct_matrix[0, :] *= math.sqrt(1.0 / N)  # u=0: DC component
        dct_matrix[1:, :] *= math.sqrt(2.0 / N)  # u>0: AC components
        
        # Apply 2D DCT: DCT_matrix @ x @ DCT_matrix^T
        x_dct = torch.matmul(dct_matrix, x)
        x_dct = torch.matmul(x_dct, dct_matrix.t())
        
        return x_dct
    
    def _idct_2d(self, x_dct: torch.Tensor) -> torch.Tensor:
        """Apply 2D inverse DCT-III transform to input tensor.
        
        Args:
            x_dct: DCT coefficients tensor of shape (..., height, width).
        
        Returns:
            Reconstructed tensor of same shape.
        """
        # Inverse DCT-III: transpose of DCT-II with same normalization
        N = x_dct.shape[-1]
        device = x_dct.device
        dtype = x_dct.dtype
        
        # Create inverse DCT matrix (same as forward, applied as transpose)
        i = torch.arange(N, device=device, dtype=dtype).unsqueeze(0)  # (1, N)
        u = torch.arange(N, device=device, dtype=dtype).unsqueeze(1)  # (N, 1)
        
        # Inverse DCT matrix: same normalization as forward DCT
        idct_matrix = torch.cos(math.pi * u * (2 * i + 1) / (2 * N))
        idct_matrix[0, :] *= math.sqrt(1.0 / N)  # u=0
        idct_matrix[1:, :] *= math.sqrt(2.0 / N)  # u>0
        
        # Apply 2D inverse DCT: IDCT_matrix^T @ x_dct @ IDCT_matrix
        x = torch.matmul(idct_matrix.t(), x_dct)
        x = torch.matmul(x, idct_matrix)
        
        return x
    
    def _create_single_dct_basis_vector(
        self,
        shape: Tuple[int, int, int],
        basis_idx: int
    ) -> torch.Tensor:
        """Create a single DCT basis vector on-the-fly (memory-efficient).
        
        Args:
            shape: Image shape (channels, height, width).
            basis_idx: Flat index into the basis vector array.
        
        Returns:
            Single basis vector tensor of shape (channels, height, width).
        """
        c, h, w = shape
        block_size = self.block_size
        
        num_blocks_h = h // block_size
        num_blocks_w = w // block_size
        coeffs_per_block = block_size * block_size
        
        # Decode the flat index into (channel, block_row, block_col, dct_u, dct_v)
        ch = basis_idx // (num_blocks_h * num_blocks_w * coeffs_per_block)
        remainder = basis_idx % (num_blocks_h * num_blocks_w * coeffs_per_block)
        
        block_row = remainder // (num_blocks_w * coeffs_per_block)
        remainder = remainder % (num_blocks_w * coeffs_per_block)
        
        block_col = remainder // coeffs_per_block
        remainder = remainder % coeffs_per_block
        
        dct_u = remainder // block_size
        dct_v = remainder % block_size
        
        # Validate decoded indices
        if not (0 <= ch < c and 
                0 <= block_row < num_blocks_h and 
                0 <= block_col < num_blocks_w and
                0 <= dct_u < block_size and 
                0 <= dct_v < block_size):
            raise ValueError(
                f"Decoded indices out of bounds: ch={ch}, block_row={block_row}, "
                f"block_col={block_col}, dct_u={dct_u}, dct_v={dct_v} for shape {shape}"
            )
        
        # Create a single DCT coefficient block
        dct_block = torch.zeros(block_size, block_size, device=self.device, dtype=torch.float32)
        dct_block[dct_u, dct_v] = self.epsilon
        
        # Apply inverse DCT to get pixel-space perturbation
        pixel_block = self._idct_2d(dct_block.unsqueeze(0).unsqueeze(0))
        pixel_block = pixel_block.squeeze(0).squeeze(0)
        
        # Create full perturbation tensor
        perturbation = torch.zeros(c, h, w, device=self.device, dtype=torch.float32)
        
        # Place the block in the appropriate location
        row_start = block_row * block_size
        row_end = row_start + block_size
        col_start = block_col * block_size
        col_end = col_start + block_size
        
        perturbation[ch, row_start:row_end, col_start:col_end] = pixel_block
        
        return perturbation
    
    def _get_dct_basis_vectors(self, shape: Tuple[int, int, int]) -> torch.Tensor:
        """Generate DCT basis vectors for the given image shape.
        
        Args:
            shape: Image shape (channels, height, width).
        
        Returns:
            Tensor of shape (num_basis_vectors, channels, height, width) containing
            all DCT basis vectors. Each basis vector is a perturbation in pixel space
            corresponding to a single DCT coefficient.
        """
        c, h, w = shape
        block_size = self.block_size
        
        # Check cache
        if (self._dct_basis_cache is not None and 
            self._dct_basis_shape == shape):
            return self._dct_basis_cache
        
        # Calculate number of blocks
        num_blocks_h = h // block_size
        num_blocks_w = w // block_size
        
        # Total number of basis vectors: channels * num_blocks * (block_size^2)
        num_basis = c * num_blocks_h * num_blocks_w * (block_size * block_size)
        
        # Initialize basis vectors tensor
        basis_vectors = torch.zeros(
            num_basis, c, h, w, 
            device=self.device, 
            dtype=torch.float32
        )
        
        # Generate each basis vector
        basis_idx = 0
        for ch in range(c):
            for block_row in range(num_blocks_h):
                for block_col in range(num_blocks_w):
                    for dct_u in range(block_size):
                        for dct_v in range(block_size):
                            # Create a single DCT coefficient block
                            dct_block = torch.zeros(block_size, block_size, device=self.device)
                            dct_block[dct_u, dct_v] = self.epsilon
                            
                            # Apply inverse DCT to get pixel-space perturbation
                            pixel_block = self._idct_2d(dct_block.unsqueeze(0).unsqueeze(0))
                            pixel_block = pixel_block.squeeze(0).squeeze(0)
                            
                            # Place the block in the appropriate location
                            row_start = block_row * block_size
                            row_end = row_start + block_size
                            col_start = block_col * block_size
                            col_end = col_start + block_size
                            
                            basis_vectors[basis_idx, ch, row_start:row_end, col_start:col_end] = pixel_block
                            basis_idx += 1
        
        # Cache the result
        self._dct_basis_cache = basis_vectors
        self._dct_basis_shape = shape
        
        return basis_vectors
    
    def _generate_dct_candidate_indices(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """Generate candidate indices in DCT space.
        
        Args:
            x: Single image tensor of shape (channels, height, width).
        
        Returns:
            Tensor of candidate indices of shape (num_candidates,) where
            each element is a flat index into the basis vector array.
        """
        c, h, w = x.shape
        block_size = self.block_size
        
        # Warn if dimensions aren't divisible by block_size
        if h % block_size != 0 or w % block_size != 0:
            import warnings
            warnings.warn(
                f"Image dimensions ({h}, {w}) are not divisible by block_size {block_size}. "
                f"Only using {h // block_size}x{w // block_size} blocks, "
                f"ignoring {h % block_size}x{w % block_size} pixels."
            )
        
        num_blocks_h = h // block_size
        num_blocks_w = w // block_size
        coeffs_per_block = block_size * block_size
        
        # Generate all candidate indices
        candidates = []
        for ch in range(c):
            for block_row in range(num_blocks_h):
                for block_col in range(num_blocks_w):
                    for dct_u in range(block_size):
                        for dct_v in range(block_size):
                            # Calculate flat index into basis vector array
                            basis_idx = (
                                ch * num_blocks_h * num_blocks_w * coeffs_per_block +
                                block_row * num_blocks_w * coeffs_per_block +
                                block_col * coeffs_per_block +
                                dct_u * block_size +
                                dct_v
                            )
                            candidates.append(basis_idx)
        
        return torch.tensor(candidates, device=self.device, dtype=torch.long)
    
    def _generate_pixel_candidate_indices(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """Generate candidate indices in pixel space (memory-efficient).
        
        Args:
            x: Single image tensor of shape (channels, height, width).
        
        Returns:
            Tensor of candidate indices of shape (num_candidates, 3) where
            each row is (channel, row, col).
        """
        c, h, w = x.shape
        
        # Generate all (channel, row, col) combinations
        channels = torch.arange(c, device=self.device)
        rows = torch.arange(h, device=self.device)
        cols = torch.arange(w, device=self.device)
        
        # Create meshgrid and flatten
        ch_grid, row_grid, col_grid = torch.meshgrid(
            channels, rows, cols, indexing='ij'
        )
        
        # Stack and reshape to (num_candidates, 3)
        candidate_indices = torch.stack([
            ch_grid.flatten(),
            row_grid.flatten(),
            col_grid.flatten()
        ], dim=1)
        
        return candidate_indices
    
    def _create_perturbation(
        self,
        shape: Tuple[int, int, int],
        candidate_idx: torch.Tensor
    ) -> torch.Tensor:
        """Create a single perturbation tensor from candidate index.
        
        Args:
            shape: Image shape (channels, height, width).
            candidate_idx: Index tensor. 
                - If use_dct=False: shape (3,) with (channel, row, col)
                - If use_dct=True: scalar tensor with flat index into basis vector array
        
        Returns:
            Perturbation tensor of the given shape.
        
        Raises:
            ValueError: If candidate_idx has invalid shape or indices out of bounds.
        """
        if self.use_dct:
            # DCT mode: candidate_idx is a flat index into basis vector array
            if candidate_idx.dim() != 0 and candidate_idx.shape != (1,):
                raise ValueError(
                    f"Expected scalar candidate_idx for DCT mode, got shape {candidate_idx.shape}"
                )
            
            # Compute basis vector on-the-fly (memory-efficient)
            basis_idx = candidate_idx.item() if candidate_idx.dim() == 0 else candidate_idx[0].item()
            return self._create_single_dct_basis_vector(shape, basis_idx)
        else:
            # Pixel mode: candidate_idx is (channel, row, col)
            if candidate_idx.shape != (3,):
                raise ValueError(f"Expected candidate_idx of shape (3,), got {candidate_idx.shape}")
            
            c, h, w = shape
            perturbation = torch.zeros(c, h, w, device=self.device)
            
            ch, row, col = candidate_idx[0].item(), candidate_idx[1].item(), candidate_idx[2].item()
            
            # Validate indices are within bounds
            if not (0 <= ch < c and 0 <= row < h and 0 <= col < w):
                raise ValueError(
                    f"Index out of bounds: ({ch}, {row}, {col}) for shape ({c}, {h}, {w})"
                )
            
            perturbation[ch, row, col] = self.epsilon
            
            return perturbation
    
    def _is_misclassified(
        self,
        x: torch.Tensor,
        y_true: torch.Tensor
    ) -> bool:
        """Check if image is misclassified.

        Args:
            x: Image tensor of shape (1, channels, height, width).
            y_true: True label tensor of shape (1,).

        Returns:
            True if misclassified, False otherwise.
        """
        with torch.no_grad():
            logits = self.model(x)
            prediction = torch.argmax(logits, dim=1)
            return (prediction != y_true).item()

    def _is_target_class(
        self,
        x: torch.Tensor,
        target_class: torch.Tensor
    ) -> bool:
        """Check if image is classified as the target class.

        Args:
            x: Image tensor of shape (1, channels, height, width).
            target_class: Target class tensor of shape (1,).

        Returns:
            True if classified as target class, False otherwise.
        """
        with torch.no_grad():
            logits = self.model(x)
            prediction = torch.argmax(logits, dim=1)
            return (prediction == target_class).item()
    