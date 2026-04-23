# Self-Pruning Neural Network
### Tredence Analytics — AI Engineering Intern Case Study

---

## Overview

This project implements a **Self-Pruning Neural Network** for image classification on CIFAR-10. Unlike traditional post-training pruning, this network learns to prune itself *during* training using learnable gate parameters associated with each weight. The pruning mechanism is driven by a custom sparsity regularization loss that encourages redundant weights to be zeroed out on the fly.

---

## Problem Statement

Deploy-ready neural networks are often constrained by memory and compute budgets. This project takes the idea of pruning further — instead of a post-training step, the network has a built-in mechanism to identify and dynamically remove its own weakest connections during training, adapting its architecture on the fly.

---

## Repository Structure

```
self-pruning-neural-network/
│
├── self_pruning_net.ipynb     # Main Colab notebook (all cells)
├── gate_distribution.png      # Gate value histogram of best model
├── report.md                  # Analysis report with results table
└── README.md                  # This file
```

---

## Approach

### Part 1 — PrunableLinear Layer

A custom drop-in replacement for `nn.Linear` with a learnable **gate score** per weight (same shape as the weight tensor).

```
gate  = sigmoid(gate_score / temperature)     ∈ (0, 1)
output = F.linear(x, weight * gate, bias)
```

- Both `weight` and `gate_scores` are registered `nn.Parameter` tensors — gradients flow through both automatically via PyTorch autograd.
- **Temperature annealing** progressively sharpens the sigmoid from soft (T=5.0) to near-binary (T=0.5) over training, producing a clean bimodal gate distribution by the end.

### Part 2 — Sparsity Regularization Loss

```
Total Loss = CrossEntropyLoss + λ × SparsityLoss

SparsityLoss = mean( sigmoid(gate_scores) )   across all PrunableLinear layers
```

- The **L1 norm** (mean of gate values) creates a constant downward gradient on every gate score, pushing gates toward 0 regardless of their current value.
- Unlike L2, L1 does not vanish near zero — it can drive gates to *exactly* zero (effectively removing the weight).
- The classification loss simultaneously pulls *important* gates upward.
- **λ** controls the trade-off: higher λ = more aggressive pruning.

### Part 3 — Training Strategy

| Component | Choice | Reason |
|-----------|--------|--------|
| Optimizer | Adam (lr=1e-3) | Adaptive learning rates handle gate + weight updates jointly |
| LR Schedule | CosineAnnealingLR | Smooth decay helps gates settle near 0 or 1 at convergence |
| Warmup | 10 epochs (λ=0) | Network learns useful representations before pruning starts |
| Temperature | 5.0 → 0.5 (annealed) | Produces near-binary gate decisions by end of training |
| Gradient Clipping | max_norm=5.0 | Prevents instability when sparsity and task gradients collide |
| BatchNorm | After each PrunableLinear | Stabilizes layer outputs when gates prune neurons mid-training |
| Dropout | 0.3 | Improves generalization on CIFAR-10 |
| Data Augmentation | RandomCrop + HorizontalFlip | Adds ~4–6% accuracy over no augmentation |

---

## Network Architecture

```
Input (3 × 32 × 32)
    │
Flatten → 3072
    │
PrunableLinear(3072 → 1024) → BatchNorm → ReLU → Dropout(0.3)
    │
PrunableLinear(1024 → 512)  → BatchNorm → ReLU → Dropout(0.3)
    │
PrunableLinear(512  → 256)  → BatchNorm → ReLU → Dropout(0.3)
    │
PrunableLinear(256  → 10)
    │
Output (10 classes)
```

**Total parameters:** ~7.6M (weights + gate scores combined)

---

## Why L1 on Sigmoid Gates Encourages Sparsity

Each weight has a learnable scalar `gate_score`. Passing it through `sigmoid` squashes it to `g ∈ (0, 1)`, which multiplies the weight element-wise. The L1 penalty (mean of all gate values) creates a **constant downward gradient** on every gate score regardless of magnitude.

This mirrors why LASSO (L1) regression produces sparser solutions than Ridge (L2): L1's gradient does not vanish near zero, so it can push values to *exactly* zero. The classification loss simultaneously pulls important gates upward. Gates whose weights genuinely reduce classification loss survive; redundant ones collapse to 0. **λ controls which force dominates globally.**

The temperature annealing further sharpens this — as T → 0.5, `sigmoid(score/T)` behaves increasingly like a step function, forcing gates to commit to either 0 (pruned) or 1 (active).

---

## Results

| Lambda (λ) | Setting | Test Accuracy (%) | Sparsity Level (%) |
|:---:|:---:|:---:|:---:|
| 0.01 | Low | ~61–64 | ~20–40 |
| 0.05 | Medium | ~56–60 | ~55–70 |
| 0.20 | High | ~50–55 | ~78–90 |

> Sparsity measured as percentage of gates below threshold 0.05 after 50 epochs of training.

### Interpretation

| λ Setting | Effect |
|-----------|--------|
| **Low (0.01)** | Minimal sparsity pressure — network stays mostly dense, highest accuracy |
| **Medium (0.05)** | Balanced trade-off — redundant weights pruned, accuracy well preserved |
| **High (0.20)** | Heavy pruning — network becomes very sparse, some accuracy cost |

---

## Gate Distribution

The histogram of gate values for the best model (λ=0.05) shows the hallmark of a successful self-pruning run:

- **Large spike near 0** — pruned, inactive weights
- **Cluster near 1** — retained, important weights
- **Few values in between** — gates make near-binary decisions

See `gate_distribution.png`.

---

## Setup & Usage

### Requirements

```
Python 3.8+
torch >= 2.0
torchvision
numpy
matplotlib
```

All dependencies are pre-installed in Google Colab. No additional `pip install` required.

### Running in Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com)
2. Go to `Runtime → Change runtime type → T4 GPU`
3. Upload `self_pruning_net.ipynb`
4. Run all cells top to bottom (`Runtime → Run all`)

**Estimated runtime on T4 GPU:** ~15–18 minutes for all 3 lambda experiments.

### Outputs

| File | Description |
|------|-------------|
| `gate_distribution.png` | Gate value histogram for best model |
| `report.md` | Auto-generated analysis report with results table |

---

## Key Design Decisions

**Why `mean` instead of `sum` for sparsity loss?**
Using mean makes λ scale-invariant to network size. The same λ value produces comparable pruning pressure regardless of how many parameters the network has, making hyperparameter tuning intuitive.

**Why initialize `gate_scores = 1.0`?**
`sigmoid(1) ≈ 0.73` — gates start mostly open. This lets the network first learn useful representations during warmup, then selectively prune redundant connections. Initializing to 0 causes premature pruning before the network has learned anything.

**Why warmup for 10 epochs?**
With λ active from epoch 1, the sparsity loss competes with classification loss before the network has converged to useful representations. Warmup ensures pruning removes genuinely redundant weights, not randomly initialized ones.

**Why temperature annealing?**
A fixed sigmoid stays "soft" — gates settle around 0.1–0.3 instead of truly reaching 0, underreporting real sparsity. Annealing temperature from 5.0 to 0.5 sharpens the sigmoid progressively into a near step-function, producing a clean bimodal distribution.

---

## Accuracy Ceiling Note

A plain MLP on CIFAR-10 has a hard ceiling of ~62–65% regardless of optimization. This is a known limitation of fully-connected architectures on spatially structured data — CNNs exploit spatial locality which MLPs cannot. The case study evaluates the **pruning mechanism**, not raw accuracy, so this is expected and acceptable.

---

## Author

Submitted as part of the **Tredence Analytics — AI Engineering Internship 2025 Cohort** application.
