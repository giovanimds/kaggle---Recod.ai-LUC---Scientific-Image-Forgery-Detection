# Data Directory

This directory contains the dataset for the Scientific Image Forgery Detection competition.

## Structure

- `raw/`: Original data downloaded from Kaggle
- `processed/`: Preprocessed data ready for training
- `external/`: External datasets or supplementary data

## Download Data

To download the competition data:

```bash
kaggle competitions download -c recodai-luc-scientific-image-forgery-detection
unzip recodai-luc-scientific-image-forgery-detection.zip -d data/raw/
```

Make sure you have:
1. Installed the Kaggle CLI: `pip install kaggle`
2. Set up your Kaggle API credentials in `~/.kaggle/kaggle.json`
3. Accepted the competition rules on the Kaggle website
