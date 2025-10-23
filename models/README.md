# Models Directory

This directory contains saved model checkpoints and trained models.

## Structure

Models will be saved with the following naming convention:
- `model_{name}_epoch_{epoch}.pth`: Model checkpoint at specific epoch
- `model_{name}_best.pth`: Best model based on validation performance

## Usage

```python
from forgery_detection.model import create_model
from forgery_detection.train import load_checkpoint

# Load a saved model
model = create_model(model_name="efficientnet_b0")
load_checkpoint(model, None, "models/model_efficientnet_b0_best.pth")
```
