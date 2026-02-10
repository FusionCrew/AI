
# Hesitation Learning Module

This folder contains the complete, self-contained environment for the Hesitation Detection deep learning model.

## Folder Structure
- `data/`: Contains the DAiSEE dataset and cached features.
- `models/`: Contains trained model (`.joblib`) and scaler files.
- `run_training.py`: Script to train the model.
- `run_feature_extraction.py`: Script to extract features from videos.
- `run_server.py`: Script to run the inference API server.
- `config.py`: Configuration settings.

## How to Run

**Note:** Run these commands from the `AI` directory (the parent directory of `hesitationLearning`).

### 1. Feature Extraction (Optional)
If you need to re-extract features from the dataset:
```bash
python -m hesitationLearning.extract_features --force
# or simply
python hesitationLearning/run_feature_extraction.py
```

### 2. Training the Model
To train the model with SMOTE and custom threshold:
```bash
python -m hesitationLearning.train --max-samples 500 --threshold 0.3
# or
python hesitationLearning/run_training.py
```

### 3. Running the Server
To start the inference API:
```bash
python hesitationLearning/run_server.py
```

## Requirements
Ensure you have the necessary packages installed:
```bash
pip install -r ../requirements.txt
```
