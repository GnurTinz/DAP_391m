# Probabilistic Palmprint Embedding with Generative Regularization for Open-Set Identification

This repository contains the official implementation for the paper **"Probabilistic Palmprint Embedding with Generative Regularization for Open-Set Identification"**.

## Authors
- **Tung Le** (tungln4@fpt.edu.vn) *
- **Thu Le** (thulvm@fpt.edu.vn)
- **Lam N.D.B** (lam01662052827@gmail.com) †
- **Tin Tran Trung** (trungtin1218@gmail.com) †
- **Thinh Nguyen Cong** (nguyencongthinh17122006@gmail.com) †
- **Thang Vo Hieu** (thanhtuan21062000@gmail.com) †

*FPT University, Ho Chi Minh City, Vietnam*  
(* Corresponding author, † Equal contribution)

---

## Abstract

Open-set palmprint identification must identify enrolled subjects from imperfect images and reject impostors. We propose modeling the Region of Interest (ROI) of each palmprint as a **diagonal Gaussian distribution**:
- **Mean ($\boldsymbol{\mu}$):** Serves as the deterministic identity representation.
- **Variance ($\boldsymbol{\sigma}^2$):** Provides an input-dependent estimate of aleatoric uncertainty.

During training, an auxiliary **U-Net decoder** is conditioned on a stochastic sample from this distribution. This generative branch encourages the latent representation to retain spatial palm-line structure and is dropped at inference time. The training loss comprises angular-margin classification, covariance regularization, image reconstruction, and heteroscedastic uncertainty calibration.

For open-set decisions, we consider similarity, probe uncertainty, and symmetric KL divergence together. We evaluate four inference modes on seven backbone-dataset combinations. Direct matching in the **latent-mean space** proves most effective, significantly improving average open-set Rank-1 and reducing average Equal Error Rate (EER).

---

## Key Contributions

1. **Probabilistic Embedding:** Palmprint features are modeled as diagonal Gaussian embeddings, explicitly decoupling the mean, variance, stochastic sample, projection head, and reconstruction decoder.
2. **Generative Regularization & Training Schedule:** The representation is trained using identity, covariance, reconstruction, and uncertainty-calibration objectives under a staged schedule.
3. **Open-Set Inference Modes:** We define four inference modes and a validation-calibrated rejection rule combining similarity, probe uncertainty, and symmetric distribution distance.
4. **Comprehensive Evaluation:** Results are reported for seven backbone-dataset combinations across three datasets (Own, Tongji, IITD) using architectures like CCNet, ResNet18, and PalmNet.

---

## Method Overview

### 1. Architecture
An encoder predicts Gaussian parameters $(\boldsymbol{\mu}, \boldsymbol{\sigma}^2)$. The mean $\boldsymbol{\mu}$ is projected through an MLP for identity learning (angular-margin classification). A stochastic sample $\mathbf{z}$ conditions a U-Net decoder through Feature-wise Linear Modulation (FiLM) to reconstruct the image. The reconstruction branch is removed at inference time.

### 2. Inference Modes
We evaluate four scoring procedures for matching a probe $p$ against a gallery template $g$:
- **M0 (Projected-mean matching):** Scored using the projected vectors.
- **M1 (Projected-space adaptation):** Adapts a residual $\boldsymbol{\delta}_r$ in the projected space before scoring.
- **M2 (Latent-space adaptation):** Adapts a residual $\boldsymbol{\delta}_\mu$ in the latent space, penalizing the KL divergence from the original probe distribution.
- **M3 (Latent-mean matching):** Direct cosine similarity between the latent means $\boldsymbol{\mu}_p$ and $\boldsymbol{\mu}_g$.

### 3. Open-Set Rejection Rule
A probe $p$ is accepted as identity $g^*$ only if it passes three thresholds selected jointly on a validation split:
1. $U_p \leq \tau_U$ (Probe uncertainty is low)
2. $S(p, g^*) \geq \tau_S$ (Cosine similarity is high)
3. $D_{\mathrm{SKL}}(p, g^*) \leq \tau_K$ (Symmetric KL divergence is low)

---

## Datasets

Experiments were conducted on the following datasets, split into training, validation, enrolled-test, and unknown-test partitions:
- **Own Dataset** (Evaluated on CCNet, ResNet18, PalmNet)
- **Tongji** (Evaluated on CCNet, ResNet18)
- **IITD** (Evaluated on CCNet, ResNet18)

---

## Project Structure

```text
.
├── config/             # Configuration files (Hydra YAMLs for datasets, models, training)
├── data/               # Datasets and gallery caches
├── logs/               # Output logs, tensorboard, and model checkpoints
├── src/                # Source code (models, losses, dataset loaders)
├── tools/              # Scripts for training and evaluation
└── implement-idea/     # Notes, diagrams, and detailed experiment scripts
```

---

## How to Run

### Training
To train the model, run the PyTorch Lightning training script in the `tools` directory. You can specify the dataset and configuration using Hydra syntax:

```bash
# Example: Training with IITD dataset
python tools/train_lightning.py dataset=iitd_hand
```
Checkpoints and logs will automatically be saved to the `logs/` directory under a versioned folder.

### Inference / Evaluation
The evaluation script extracts features, builds the gallery, and computes EER, Rank-1 Accuracy, and other metrics automatically.

For baseline evaluation (M0: Projected-mean matching):
```bash
# Define your checkpoint path and dataset
CKPT="logs/Unet_Palmnet/version_2/checkpoints/last.ckpt"
DATASET="iitd_hand"

# Run M0 evaluation
python tools/eval_attendance.py checkpoint=$CKPT dataset=$DATASET +eval=eval eval.mode=0
```

For advanced evaluation (M2: Latent-space adaptation with spherical rotation):
```bash
python tools/eval_attendance.py checkpoint=$CKPT dataset=$DATASET +eval=eval eval.mode=2 eval.neg_strategy=spherical
```

*Note: The first time you run an evaluation for a checkpoint, it will extract and cache the features. Subsequent runs with different modes will be much faster. You can force re-extraction by appending `eval.force_reextract=true`.*

---
**Keywords:** *Palmprint recognition, probabilistic embedding, open-set identification, aleatoric uncertainty, generative regularization*
