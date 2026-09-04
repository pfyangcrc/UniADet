<p align="center"> <b>EN</b> | <a href="README-zh.md">ZH</a></p>

# UniADet
Unofficial PyTorch re-implementation of UniADet.

## Project Usage Guide

### 1. Environment Setup
Install dependencies:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 2. Training

1. Configure training settings in `config_train.yaml`
2. Start training:

```
python train.py
```

Trained model weights will be saved after training finishes.

### 3. Evaluation / Inference

1. Configure inference and evaluation settings in `config_infer.yaml`
2. Run evaluation:

```
python infer.py
```

The script loads the trained model and outputs evaluation results on the test set.

## File Description

- `config_train.yaml`: Configuration for training
- `config_infer.yaml`: Configuration for inference/evaluation
- `train.py`: Training entry script
- `infer.py`: Inference and evaluation entry script

## Notes

- Training and inference use separate configuration files.
- Make sure the model weight path in `config_infer.yaml` matches the saved training checkpoint path.
- Training hyperparameters, network settings and augmentation strategies are defined in `config_train.yaml`.
