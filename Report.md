# Self-Pruning Neural Network on CIFAR-10
### Tredence Analytics — AI Engineering Internship 2025 | Case Study Submission

---

## Problem Statement

Deploying large neural networks is constrained by memory and compute budgets. Traditional pruning removes unimportant weights **after** training. This project goes further — the network learns to prune itself **during** training.

**Core mechanism:** Each weight is associated with a learnable scalar **gate parameter**. This gate, passed through a sigmoid, produces a value in `(0, 1)` that multiplies the weight's output. A gate approaching 0 effectively removes the weight. A custom sparsity loss drives unimportant gates toward zero throughout training — no post-training pruning step required.

**Dataset:** CIFAR-10 — 10 classes, 50,000 training / 10,000 test images (3 × 32 × 32).

---

### Why L1 and Not L2

| Property | L1 — used here | L2 — not used |
|----------|----------------|----------------|
| Gradient magnitude near zero | **Constant — does not vanish** | Vanishes proportionally to value |
| Ability to reach exactly zero | **Yes** | No — only approaches asymptotically |
| Induces true sparsity? | **Yes** | No — only shrinks values uniformly |

This is exactly the same reason **LASSO (L1) regression** produces truly sparse solutions while **Ridge (L2) regression** only shrinks coefficients without zeroing them.

---

## Results Table

> This section directly addresses the second required report item from the case study specification.

Training was run for three values of λ — low, medium, and high — across 50 epochs each (10 warmup + 40 pruning) on a Google Colab T4 GPU.

| Lambda (λ) | Setting | Test Accuracy (%) | Sparsity Level (%) |
|:---:|:---:|:---:|:---:|
| 0.1 | Low | 60.85 | 77.93 |
| 0.5 | Medium (Best) | 60.31 | 84.99 |
| 2.0 | High | 60.17 | 92.25 |

---

## Gate Value Distribution

> This section directly addresses the third required report item from the case study specification.

The histogram below shows the distribution of final gate values — `sigmoid(gate_score)` — for all weights in the best model (λ=0.5) after 50 epochs of training.

### How to Read This Plot

| Feature in the Plot | What It Means |
|---------------------|---------------|
| **Large spike near 0** | These gates have been driven to near-zero by the sparsity loss. Their corresponding weights are effectively pruned — removed from the network's computation |
| **Cluster near 1** | These gates are held up by the classification loss. Their weights are genuinely important for CIFAR-10 prediction and survived pruning pressure |
| **Near-empty region in between** | Gates do not remain ambiguous — they commit to one side. This binary polarization is the hallmark of a successful self-pruning mechanism |
