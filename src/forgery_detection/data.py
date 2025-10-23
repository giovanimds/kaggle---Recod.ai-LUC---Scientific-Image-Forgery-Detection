"""
Data Loading Utilities
=======================

Module for loading and preprocessing images for the forgery detection task.
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ForgeryDetectionDataset(Dataset):
    """
    Dataset class for loading scientific images for forgery detection.
    
    Args:
        data_dir: Path to the directory containing images
        transform: Optional torchvision transforms to apply
        split: Dataset split ('train', 'val', or 'test')
    """
    
    def __init__(
        self,
        data_dir: str,
        transform: Optional[transforms.Compose] = None,
        split: str = "train"
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.split = split
        self.images = []
        self.labels = []
        
        # Load image paths and labels
        self._load_data()
    
    def _load_data(self):
        """Load image paths and labels from the data directory."""
        # This will be implemented based on the actual data structure
        # from the Kaggle competition
        pass
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single image and its label.
        
        Args:
            idx: Index of the item
            
        Returns:
            Tuple of (image_tensor, label)
        """
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(split: str = "train", img_size: int = 224) -> transforms.Compose:
    """
    Get data augmentation transforms for training or validation.
    
    Args:
        split: Dataset split ('train' or 'val')
        img_size: Target image size
        
    Returns:
        Composed transforms
    """
    if split == "train":
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
