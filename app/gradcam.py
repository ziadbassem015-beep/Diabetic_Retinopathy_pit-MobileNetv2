"""
Grad-CAM utilities for the hybrid MobileNetV2 + PiT model.
"""

from typing import Optional, Tuple

import cv2
import numpy as np
import torch


def generate_gradcam_overlay(
    model_manager,
    input_tensor: torch.Tensor,
    display_image: np.ndarray,
    class_idx: Optional[int] = None,
    alpha: float = 0.45,
) -> Tuple[np.ndarray, int, float]:
    """
    Generate a Grad-CAM overlay from the MobileNetV2 convolutional branch.

    Args:
        model_manager: Loaded ModelManager instance.
        input_tensor: Preprocessed tensor with shape (1, 3, 224, 224).
        display_image: Preprocessed RGB image array with shape (224, 224, 3).
        class_idx: Optional class index to explain. Uses predicted class if omitted.
        alpha: Heatmap opacity in the overlay.

    Returns:
        Tuple of (RGB overlay image, explained class index, class confidence).
    """
    model = model_manager.model
    target_layer = model.Model_Mobile.features[-1]
    activations = {}
    gradients = {}

    def forward_hook(_module, _inputs, output):
        activations["value"] = output.detach()

    def backward_hook(_module, _grad_input, grad_output):
        gradients["value"] = grad_output[0].detach()

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad(set_to_none=True)
        tensor = input_tensor.to(model_manager.device)

        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)

        if class_idx is None:
            class_idx = int(probabilities.argmax(dim=1).item())

        confidence = float(probabilities[0, class_idx].detach().cpu().item())
        score = logits[:, class_idx].sum()
        score.backward()

        if "value" not in activations or "value" not in gradients:
            raise RuntimeError("Grad-CAM hooks did not capture model activations")

        activation = activations["value"][0]
        gradient = gradients["value"][0]
        weights = gradient.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=0))

        cam_min = cam.min()
        cam_max = cam.max()
        if torch.isclose(cam_max, cam_min):
            cam = torch.zeros_like(cam)
        else:
            cam = (cam - cam_min) / (cam_max - cam_min)

        cam_np = cam.detach().cpu().numpy()
        cam_np = cv2.resize(
            cam_np,
            (display_image.shape[1], display_image.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )

        heatmap = cv2.applyColorMap(np.uint8(255 * cam_np), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        overlay = cv2.addWeighted(
            display_image.astype(np.uint8),
            1.0 - alpha,
            heatmap.astype(np.uint8),
            alpha,
            0,
        )

        return overlay, class_idx, confidence

    finally:
        forward_handle.remove()
        backward_handle.remove()
        model.zero_grad(set_to_none=True)
