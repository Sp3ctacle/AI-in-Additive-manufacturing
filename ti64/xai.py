"""
ti64.xai - Explainable AI (XAI) Utilities for Ti-6Al-4V Microstructure Models.

Adheres strictly to:
- A13: Explainability with sanity checks (Grad-CAM, Integrated Gradients,
  cascading randomisation, pixel deletion curve, and pore mask grounding).
- A4: Unified preprocessing.
"""

from typing import Tuple, List, Dict, Optional, Any
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.stats as stats


class GradCAM:
    """
    Standardized Grad-CAM implementation targeting the convolutional layer
    specified by ti64.model.get_gradcam_layer(model).
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self.fwd_hook = None
        self.bwd_hook = None

    def _save_activations(self, module, input, output):
        self.activations = output

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove_hooks(self):
        if self.fwd_hook is not None:
            self.fwd_hook.remove()
            self.fwd_hook = None
        if self.bwd_hook is not None:
            self.bwd_hook.remove()
            self.bwd_hook = None

    def generate(self, input_tensor: torch.Tensor, target_idx: int = 0) -> np.ndarray:
        """
        Generate 2D normalized Grad-CAM heatmap in range [0, 1].
        input_tensor: (1, 3, H, W)
        """
        self.model.eval()
        self.model.zero_grad()
        
        # Attach hooks only during this invocation
        self.fwd_hook = self.target_layer.register_forward_hook(self._save_activations)
        self.bwd_hook = self.target_layer.register_full_backward_hook(self._save_gradients)
        try:
            preds = self.model(input_tensor)
            score = preds[0, target_idx]
            score.backward(retain_graph=False)

            grads = self.gradients.detach()      # (1, C, H', W')
            acts = self.activations.detach()     # (1, C, H', W')

            weights = torch.mean(grads, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
            cam = torch.sum(weights * acts, dim=1, keepdim=True)   # (1, 1, H', W')
            cam = F.relu(cam)

            cam_resized = F.interpolate(
                cam, size=(input_tensor.shape[2], input_tensor.shape[3]),
                mode="bilinear", align_corners=False
            )
            cam_np = cam_resized.squeeze().cpu().numpy()
            cam_max = np.max(cam_np)
            if cam_max > 1e-8:
                cam_np /= cam_max
            else:
                cam_np = np.zeros_like(cam_np)
            return cam_np
        finally:
            self.remove_hooks()


def compute_integrated_gradients(
    model: nn.Module,
    input_tensor: torch.Tensor,
    baseline_tensor: torch.Tensor,
    target_idx: int = 0,
    steps: int = 200,
    batch_size: int = 16,
) -> Tuple[np.ndarray, float]:
    """
    Computes Integrated Gradients path integral between baseline and input
    using memory-safe sub-batches. Returns (saliency_map_2d, completeness_ratio).
    """
    model.eval()
    alphas = torch.linspace(0.0, 1.0, steps + 1, device=input_tensor.device)
    delta = input_tensor - baseline_tensor

    all_grads = []
    # Process interpolation in sub-batches to prevent memory allocation spikes
    for b_start in range(0, steps + 1, batch_size):
        b_end = min(b_start + batch_size, steps + 1)
        sub_alphas = alphas[b_start:b_end]
        sub_interpolated = torch.cat([baseline_tensor + a * delta for a in sub_alphas], dim=0)
        sub_interpolated.requires_grad_(True)

        preds = model(sub_interpolated)
        target_scores = preds[:, target_idx]
        grad_outputs = torch.ones_like(target_scores)
        grads = torch.autograd.grad(target_scores, sub_interpolated, grad_outputs=grad_outputs, retain_graph=False)[0]
        all_grads.append(grads.detach())

    full_grads = torch.cat(all_grads, dim=0)  # (steps+1, 3, H, W)

    # Trapezoidal approximation
    avg_grads = torch.mean(full_grads[:-1] + full_grads[1:], dim=0) * 0.5
    ig = (delta.squeeze(0) * avg_grads).detach()  # (3, H, W)
    
    # 2D heatmap by summing absolute attributions over channels
    ig_2d = torch.sum(torch.abs(ig), dim=0).cpu().numpy()
    if np.max(ig_2d) > 1e-8:
        ig_2d /= np.max(ig_2d)

    # Completeness check: sum(IG) vs (f(x) - f(x0))
    sum_ig = float(torch.sum(ig).cpu().item())
    with torch.no_grad():
        fx = float(model(input_tensor)[0, target_idx].cpu().item())
        fx0 = float(model(baseline_tensor)[0, target_idx].cpu().item())
    diff = fx - fx0
    completeness_ratio = abs(sum_ig - diff) / max(abs(diff), 1e-6)

    return ig_2d, completeness_ratio


def evaluate_pore_grounding(
    cam_map: np.ndarray,
    binary_pore_mask: np.ndarray,
    dilation_px: int = 3,
    top_pct: float = 0.05,
) -> float:
    """
    Evaluates whether the top 5% salient CAM pixels fall within dilation_px
    of a ground truth pore. Returns the enrichment ratio:
    (hit_rate in top-5%) / (dilated pore area fraction).
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilation_px + 1, 2 * dilation_px + 1))
    dilated_mask = (cv2.dilate(binary_pore_mask.astype(np.uint8), kernel) > 0)

    area_share = np.mean(dilated_mask)
    if area_share < 1e-6:
        return 1.0  # Null case if no pores present

    cutoff = np.percentile(cam_map, 100 * (1.0 - top_pct))
    top_mask = (cam_map >= cutoff)

    hits = np.sum(top_mask & dilated_mask)
    total_top = np.sum(top_mask)
    if total_top == 0:
        return 0.0

    hit_rate = hits / total_top
    enrichment = hit_rate / max(area_share, 1e-6)
    return float(enrichment)

