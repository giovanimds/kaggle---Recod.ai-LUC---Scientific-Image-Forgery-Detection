#!/usr/bin/env python3
"""
Training Script
===============

Main script for training the forgery detection model.

Usage:
    python scripts/train.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from forgery_detection.config import TrainingConfig
from forgery_detection.model import create_model
from forgery_detection.data import ForgeryDetectionDataset, get_transforms
from forgery_detection.train import train_epoch, evaluate, save_checkpoint


def main():
    """Main training loop."""
    # Load configuration
    config = TrainingConfig()
    
    print("=" * 60)
    print("Scientific Image Forgery Detection - Training")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Model: {config.model_name}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Device: {config.device}")
    print("=" * 60)
    
    # Set device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Create model
    print("\nCreating model...")
    model = create_model(
        model_name=config.model_name,
        num_classes=config.num_classes,
        pretrained=config.pretrained
    )
    model = model.to(device)
    
    # Create datasets and dataloaders
    print("Loading datasets...")
    try:
        train_dataset = ForgeryDetectionDataset(
            data_dir=config.data_dir + "/train",
            transform=get_transforms("train", config.img_size),
            split="train"
        )
        
        val_dataset = ForgeryDetectionDataset(
            data_dir=config.data_dir + "/val",
            transform=get_transforms("val", config.img_size),
            split="val"
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )
        
        print(f"Train dataset: {len(train_dataset)} images")
        print(f"Validation dataset: {len(val_dataset)} images")
    except Exception as e:
        print(f"\nError loading datasets: {e}")
        print("\nPlease download the competition data and place it in the data/raw directory.")
        print("Run: kaggle competitions download -c recodai-luc-scientific-image-forgery-detection")
        return
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = None
    if config.use_scheduler:
        if config.scheduler_type == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.num_epochs
            )
        elif config.scheduler_type == "step":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=10, gamma=0.1
            )
    
    # TensorBoard writer
    writer = SummaryWriter(config.log_dir)
    
    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    
    print("\nStarting training...")
    print("=" * 60)
    
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        print("-" * 60)
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, scheduler
        )
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # Log metrics
        writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
        writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
        writer.add_scalar("Accuracy/train", train_metrics["accuracy"], epoch)
        writer.add_scalar("Accuracy/val", val_metrics["accuracy"], epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)
        
        # Print metrics
        print(f"\nTrain Loss: {train_metrics['loss']:.4f}, "
              f"Train Acc: {train_metrics['accuracy']:.2f}%")
        print(f"Val Loss: {val_metrics['loss']:.4f}, "
              f"Val Acc: {val_metrics['accuracy']:.2f}%")
        
        # Save best model
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            save_path = Path(config.model_save_dir) / f"model_{config.model_name}_best.pth"
            save_checkpoint(model, optimizer, epoch, val_metrics["loss"], str(save_path))
            print(f"✓ Saved best model (Val Acc: {best_val_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if config.early_stopping_patience and patience_counter >= config.early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_path = Path(config.model_save_dir) / f"model_{config.model_name}_epoch_{epoch+1}.pth"
            save_checkpoint(model, optimizer, epoch, val_metrics["loss"], str(save_path))
    
    writer.close()
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
