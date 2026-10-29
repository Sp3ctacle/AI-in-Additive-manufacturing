"""
ti64.model - PyTorch Model Architecture Definition for Ti-6Al-4V LPBF Microstructure.

Follows Global Rules:
- Backbone from timm with num_classes=0
- Head: Dropout(0.2) -> Linear(F, 256) -> SiLU -> Dropout(0.15) -> Linear(256, 128) -> SiLU -> Dropout(0.1) -> Linear(128, n_outputs)
- get_gradcam_layer(model) returning the last feature stage and its name.
"""

from typing import Tuple
import torch
import torch.nn as nn
import timm


class Ti64Regressor(nn.Module):
    """
    Unified Ti-6Al-4V Microstructure Regression & Feature Extraction Network.
    Wraps a timm feature extractor with the standardized 3-stage MLP regression head.
    """
    def __init__(
        self,
        n_outputs: int = 1,
        backbone: str = "tf_efficientnetv2_s",
        pretrained: bool = True,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.n_outputs = n_outputs

        # Create backbone without classification head (num_classes=0 returns pooled feature vector)
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
        )

        in_features = getattr(self.backbone, "num_features", None)
        if in_features is None:
            # Fallback probe
            dummy = torch.zeros(1, 3, 224, 224)
            with torch.no_grad():
                out = self.backbone(dummy)
            in_features = out.shape[1]

        # Standardized head specification:
        # Dropout(0.2) -> Linear(F, 256) -> SiLU -> Dropout(0.15) ->
        # Linear(256, 128) -> SiLU -> Dropout(0.1) -> Linear(128, n_outputs)
        self.head = nn.Sequential(
            nn.Dropout(p=0.20),
            nn.Linear(in_features, 256),
            nn.SiLU(),
            nn.Dropout(p=0.15),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Dropout(p=0.10),
            nn.Linear(128, n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract unpooled spatial feature maps for explainability / Grad-CAM."""
        if hasattr(self.backbone, "forward_features"):
            return self.backbone.forward_features(x)
        return self.backbone(x)


def build_model(
    n_outputs: int = 1,
    backbone: str = "tf_efficientnetv2_s",
    pretrained: bool = True,
) -> nn.Module:
    """Build standardized Ti-6Al-4V regression model."""
    return Ti64Regressor(n_outputs=n_outputs, backbone=backbone, pretrained=pretrained)


def get_gradcam_layer(model: nn.Module) -> Tuple[nn.Module, str]:
    """
    Return the last convolutional feature stage module and its identifier string
    for Grad-CAM explainability analysis (Rule A13).
    """
    bb = getattr(model, "backbone", model)

    # For EfficientNet / MobileNet / ConvNeXt families in timm
    if hasattr(bb, "conv_head"):
        return bb.conv_head, "backbone.conv_head"
    if hasattr(bb, "bn2"):
        return bb.bn2, "backbone.bn2"
    if hasattr(bb, "head") and hasattr(bb.head, "fc"):
        return bb.head, "backbone.head"
    if hasattr(bb, "layer4"):  # ResNet family
        return bb.layer4[-1], "backbone.layer4[-1]"

    # Fallback to the last convolutional layer found in the backbone
    last_conv = None
    last_name = None
    for name, module in bb.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
            last_name = f"backbone.{name}"

    if last_conv is not None:
        return last_conv, last_name

    raise ValueError("Could not automatically locate a convolutional stage for Grad-CAM in this model.")
