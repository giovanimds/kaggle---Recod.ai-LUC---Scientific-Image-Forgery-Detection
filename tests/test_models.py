"""
Test suite for the forgery detection package.
"""

import pytest
import torch
import torch.nn as nn
from forgery_detection.model import (
    ForgeryDetectionModel,
    MultiTaskForgeryModel,
    create_model
)


class TestModels:
    """Test cases for model creation and forward passes."""
    
    def test_forgery_detection_model_creation(self):
        """Test that ForgeryDetectionModel can be created."""
        model = ForgeryDetectionModel(
            model_name="efficientnet_b0",
            num_classes=2,
            pretrained=False
        )
        assert isinstance(model, nn.Module)
    
    def test_forgery_detection_model_forward(self):
        """Test forward pass through ForgeryDetectionModel."""
        model = ForgeryDetectionModel(
            model_name="efficientnet_b0",
            num_classes=2,
            pretrained=False
        )
        model.eval()
        
        # Create dummy input
        batch_size = 4
        x = torch.randn(batch_size, 3, 224, 224)
        
        # Forward pass
        with torch.no_grad():
            output = model(x)
        
        # Check output shape
        assert output.shape == (batch_size, 2)
    
    def test_multi_task_model_creation(self):
        """Test that MultiTaskForgeryModel can be created."""
        model = MultiTaskForgeryModel(
            model_name="efficientnet_b0",
            num_classes=2,
            num_aux_classes=5,
            pretrained=False
        )
        assert isinstance(model, nn.Module)
    
    def test_multi_task_model_forward(self):
        """Test forward pass through MultiTaskForgeryModel."""
        model = MultiTaskForgeryModel(
            model_name="efficientnet_b0",
            num_classes=2,
            num_aux_classes=5,
            pretrained=False
        )
        model.eval()
        
        # Create dummy input
        batch_size = 4
        x = torch.randn(batch_size, 3, 224, 224)
        
        # Forward pass
        with torch.no_grad():
            main_out, aux_out = model(x)
        
        # Check output shapes
        assert main_out.shape == (batch_size, 2)
        assert aux_out.shape == (batch_size, 5)
    
    def test_create_model_single_task(self):
        """Test create_model factory function for single task."""
        model = create_model(
            model_name="efficientnet_b0",
            num_classes=2,
            pretrained=False,
            multi_task=False
        )
        assert isinstance(model, ForgeryDetectionModel)
    
    def test_create_model_multi_task(self):
        """Test create_model factory function for multi-task."""
        model = create_model(
            model_name="efficientnet_b0",
            num_classes=2,
            pretrained=False,
            multi_task=True,
            num_aux_classes=5
        )
        assert isinstance(model, MultiTaskForgeryModel)


class TestDataTransforms:
    """Test cases for data transformations."""
    
    def test_get_transforms_train(self):
        """Test that training transforms can be created."""
        from forgery_detection.data import get_transforms
        
        transforms = get_transforms(split="train", img_size=224)
        assert transforms is not None
    
    def test_get_transforms_val(self):
        """Test that validation transforms can be created."""
        from forgery_detection.data import get_transforms
        
        transforms = get_transforms(split="val", img_size=224)
        assert transforms is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
