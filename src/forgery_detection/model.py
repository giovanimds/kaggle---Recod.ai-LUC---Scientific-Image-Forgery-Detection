"""
Model Definitions
==================

Module containing model architectures for forgery detection.
"""

import torch
import torch.nn as nn
import timm
from typing import Optional


class ForgeryDetectionModel(nn.Module):
    """
    Base model for forgery detection using transfer learning.
    
    Args:
        model_name: Name of the pretrained model from timm
        num_classes: Number of output classes (default: 2 for binary classification)
        pretrained: Whether to use pretrained weights
    """
    
    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        num_classes: int = 2,
        pretrained: bool = True
    ):
        super().__init__()
        
        # Load pretrained model from timm
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.backbone(x)


class MultiTaskForgeryModel(nn.Module):
    """
    Multi-task model for forgery detection with auxiliary tasks.
    
    Args:
        model_name: Name of the pretrained model from timm
        num_classes: Number of output classes
        num_aux_classes: Number of auxiliary task classes
        pretrained: Whether to use pretrained weights
    """
    
    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        num_classes: int = 2,
        num_aux_classes: Optional[int] = None,
        pretrained: bool = True
    ):
        super().__init__()
        
        # Load pretrained model
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0  # Remove classification head
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            feature_dim = features.shape[1]
        
        # Main classification head
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        # Auxiliary classification head (optional)
        self.aux_classifier = None
        if num_aux_classes is not None:
            self.aux_classifier = nn.Linear(feature_dim, num_aux_classes)
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass through the model.
        
        Returns:
            Tuple of (main_output, aux_output) or just main_output
        """
        features = self.backbone(x)
        main_out = self.classifier(features)
        
        if self.aux_classifier is not None:
            aux_out = self.aux_classifier(features)
            return main_out, aux_out
        
        return main_out


def create_model(
    model_name: str = "efficientnet_b0",
    num_classes: int = 2,
    pretrained: bool = True,
    multi_task: bool = False,
    num_aux_classes: Optional[int] = None
) -> nn.Module:
    """
    Factory function to create a model.
    
    Args:
        model_name: Name of the pretrained model
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        multi_task: Whether to create a multi-task model
        num_aux_classes: Number of auxiliary task classes (for multi-task)
        
    Returns:
        Model instance
    """
    if multi_task:
        return MultiTaskForgeryModel(
            model_name=model_name,
            num_classes=num_classes,
            num_aux_classes=num_aux_classes,
            pretrained=pretrained
        )
    else:
        return ForgeryDetectionModel(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained
        )
