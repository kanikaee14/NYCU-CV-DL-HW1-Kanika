# NYCU-CV-DL-HW1-Kanika
# NYCU Computer Vision with Deep Learning Spring 2026 HW1

**Student Id:** 314540012 
**Name:** Kanika  


## Introduction

This project implements an image classification model for the course *Special Topics in Computer Vision using Deep Learning* at NYCU.

The objective is to classify images into 100 categories using deep learning techniques.  
The backbone model used is **ResNet50**, enhanced with additional techniques to improve performance.

### Key Ideas

- Use of **ResNet50** as backbone
- Addition of **CBAM (Convolutional Block Attention Module)** for better feature attention
- Two-phase training strategy:
  - Phase 1: Backbone frozen (feature learning)
  - Phase 2: Full fine-tuning
- Advanced augmentation:
  - **MixUp**
  - **CutMix**
- Learning rate scheduling using **Warm Restarts**

---

## Environment Setup

```bash
conda create -n hw1 python=3.10
conda activate hw1

pip install torch torchvision torchaudio
pip install pillow pandas numpy
```
## Usage
### Training
```bash
python train_v2.py
```
### Inference
```bash
python inference_v2.py --model1 ./best_model.pth
```
## Performance Snapshot
```bash
Best Validation Accuracy: 90.67%
Training Strategy:
Phase 1: Warmup (frozen backbone)
Phase 2: Full fine-tuning with augmentations
```
## Dataset:
```bash
Train: 20,724 images
Validation: 300 images
Test: 2,344 images
```
The model shows strong generalization performance due to attention mechanisms and advanced augmentation strategies.

## Notes

Training performed on MacOS using MPS (Apple Silicon acceleration)
Early stopping applied at epoch 31
Prediction file formatted for Codabench submission

## References
He et al., "Deep Residual Learning for Image Recognition"

Woo et al., "CBAM: Convolutional Block Attention Module"

PyTorch Documentation
## performance Snapshot 
![Project Snapshot](https://github.com/kanikaee14/NYCU-CV-DL-HW1-Kanika/raw/main/snapshot.png)


