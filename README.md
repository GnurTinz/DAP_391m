# Probabilistic Palmprint Embedding with Generative Regularization for Open-Set Identification

This repository contains the implementation for the paper **"Probabilistic Palmprint Embedding with Generative Regularization for Open-Set Identification"**.

## Authors
- **Tung Le** (tungln4@fpt.edu.vn)
- **Thu Le** (thulvm@fpt.edu.vn)
- **Lam N.D.B** (lam01662052827@gmail.com)
- **Tin Tran Trung** (trungtin1218@gmail.com)
- **Thinh Nguyen Cong** (nguyencongthinh17122006@gmail.com)
- **Thang Vo Hieu** (thanhtuan21062000@gmail.com)

*FPT University, Ho Chi Minh City, Vietnam*

## Abstract

Open-set palmprint identification must identify enrolled subjects from imperfect images and reject subjects not in the gallery. We propose a solution where the Region of Interest (ROI) of each palmprint is modeled as a **diagonal Gaussian distribution**.
- **Mean:** Acts as the identity representation.
- **Variance:** Provides an input-dependent estimate of aleatoric uncertainty.

During training, an auxiliary **U-Net decoder** is conditioned on a stochastic sample from this distribution. This encourages the latent representation to retain information about palm-line structure and is dropped at inference time. The total training loss is composed of angular-margin classification, covariance regularization, image reconstruction, and heteroscedastic uncertainty calibration.

For open-set decisions, we consider similarity, probe uncertainty, and symmetric KL divergence together. We evaluate four inference modes on seven backbone-dataset combinations. **Latent-mean matching** emerges as the most effective inference option in almost all settings, significantly improving average open-set Rank-1 accuracy and reducing Equal Error Rate (EER).

## Key Contributions

1. **Probabilistic Embedding:** Palmprint features are modeled as diagonal Gaussian embeddings, explicitly decoupling the mean, variance, stochastic sample, projection head, and reconstruction decoder.
2. **Generative Regularization & Training Schedule:** The representation is trained with identity, covariance, reconstruction, and uncertainty-calibration objectives under a staged schedule.
3. **Open-Set Inference Modes:** We define four inference modes (M0: projected-mean matching, M1: projected-space adaptation, M2: latent-space adaptation, M3: latent-mean matching) where rejection thresholds (for similarity, uncertainty, and KL divergence) are determined via an **automated grid search** over the validation set.
4. **Comprehensive Evaluation:** Results are reported for seven backbone-dataset combinations across three datasets (Own, Tongji, IITD) using models like CCNet, ResNet18, and PalmNet.

## Method Overview

- **Architecture:** An encoder predicts Gaussian parameters $(\boldsymbol{\mu}, \boldsymbol{\sigma}^2)$. The mean $\boldsymbol{\mu}$ is projected for identity learning, while a stochastic sample $\mathbf{z}$ conditions a U-Net decoder using FiLM (Feature-wise Linear Modulation) during training.
- **Inference Rules:** Probes are accepted based on a combination of similarity score, probe uncertainty ($U_p$), and symmetric KL divergence ($D_{\mathrm{SKL}}$).
- **Matching Modes:** 
  - **M0:** Projected-mean matching
  - **M1:** Projected-space adaptation
  - **M2:** Latent-space adaptation
  - **M3:** Latent-mean matching (generally outperforms others)

## Datasets

Experiments were conducted on the following datasets:
- **Own Dataset** (Evaluated on CCNet, ResNet18, PalmNet)
- **Tongji** (Evaluated on CCNet, ResNet18)
- **IITD** (Evaluated on CCNet, ResNet18)

## Project Structure

```text
.
├── config/             # Configuration files (e.g., dataset YAMLs, hyperparameters)
├── data/               # Datasets and gallery caches
├── logs/               # Output logs and model checkpoints
├── src/                # Source code (models, dataset loaders, architectures)
├── tools/              # Scripts for training, evaluation, and utilities
└── implement-idea/     # Notes, ideas, and detailed experiment scripts
```

## How to Run

### Training
To train the model, you can run the lightning training script provided in the `tools` directory. You can specify the dataset and other parameters using Hydra syntax:

```bash
python tools/train_lightning.py dataset=iitd_hand
```
Checkpoints and logs will automatically be saved to the `logs/` directory.

### Inference / Evaluation
The evaluation script extracts features, builds the gallery, and computes EER and Rank-1 Accuracy automatically.

For baseline evaluation (Projected Space without adaptation):
```bash
# Define your checkpoint path and dataset
CKPT="logs/Unet_Palmnet/version_2/checkpoints/last.ckpt"
DATASET="iitd_hand"

# Run evaluation
python tools/eval_attendance.py checkpoint=$CKPT dataset=$DATASET +eval=eval eval.mode=0
```

For advanced evaluation involving negative mining and adaptation (e.g., Optimize in Latent Space using Spherical Rotation):
```bash
python tools/eval_attendance.py checkpoint=$CKPT dataset=$DATASET +eval=eval eval.mode=2 eval.neg_strategy=spherical
```
*Note: The first time you run an evaluation for a checkpoint, it will extract and cache the features. Subsequent runs with different modes will be much faster. You can force re-extraction by passing `eval.force_reextract=true`.*

## Keywords
*Palmprint recognition, probabilistic embedding, open-set identification, aleatoric uncertainty, generative regularization*
