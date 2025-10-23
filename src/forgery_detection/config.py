"""
Training Configuration
======================

Central configuration file for training parameters.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    
    # Model settings
    model_name: str = "efficientnet_b0"
    num_classes: int = 2
    pretrained: bool = True
    
    # Training settings
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # Data settings
    img_size: int = 224
    num_workers: int = 4
    
    # Paths
    data_dir: str = "data/raw"
    model_save_dir: str = "models"
    log_dir: str = "runs"
    
    # Device
    device: str = "cuda"
    
    # Early stopping
    early_stopping_patience: Optional[int] = 10
    
    # Scheduler
    use_scheduler: bool = True
    scheduler_type: str = "cosine"  # cosine, step, plateau
    
    # Augmentation
    use_heavy_augmentation: bool = True


@dataclass
class InferenceConfig:
    """Configuration for model inference."""
    
    model_name: str = "efficientnet_b0"
    num_classes: int = 2
    checkpoint_path: str = "models/model_best.pth"
    batch_size: int = 32
    img_size: int = 224
    device: str = "cuda"
