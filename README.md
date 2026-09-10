# Efficient JPEG Restoration in the Wavelet Domain via Mean Flows

[![arXiv](https://img.shields.io/badge/arXiv-2608.28730-b31b1b.svg)](https://arxiv.org/abs/2608.28730)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

Official PyTorch implementation of **"Efficient JPEG Restoration in the Wavelet Domain via Mean Flows"**.

---

## ⚡ Highlights

* **Ultra-Lightweight & Fast:** 65M parameters, sustaining **8.05 img/s at 1024×1024** on a single RTX 3090 (~4.9× the throughput of one-step SODiff with 1/20th the parameters).
* **Wavelet-Domain Processing:** Replaces heavy learned VAEs with an exactly invertible **two-level 2D Haar transform**, avoiding lossy latent compression and reconstruction bottlenecks.
* **Rank-Enhanced Linear-Attention DiT:** Predicts residual corrections in the wavelet domain with internal compression-severity estimation.
* **Distillation-Free 1–2 Step Inference:** Optimized with an improved **MeanFlow** objective for ultra-fast generation without multi-stage distillation pipelines.
* **State-of-the-Art Perceptual Quality:** Lowest LPIPS on standard benchmarks (**LIVE-1**, **Urban100**, and **DIV2K-val**) at moderate-to-high compression levels (QF 10 and 20).

---

## 🛠️ Setup

### Installation (TODO)

Clone the repository and install dependencies:

```bash
git clone [https://github.com/stefanasandei/jpeg-restore.git](https://github.com/stefanasandei/jpeg-restore.git)
cd jpeg-restore
TODO
```

---

## 🚀 Quickstart

### 1. Inference

Run restoration on a compressed image or directory:

```bash
TODO
```

### 2. Training

Train from scratch using the provided script:

```bash
bash ./scripts/train.sh

```

---

## 📊 Benchmark Results

| Benchmark | QF | Metric | Ours (65M) | Throughput (1024×1024) |
| --- | --- | --- | --- | --- |
| **LIVE-1 / Urban100 / DIV2K-val** | 10, 20 | **Lowest LPIPS** | **SOTA** | **8.05 img/s** (RTX 3090) |

> *Note:* Pretrained large-scale diffusion priors remain stronger under extreme compression (e.g., QF 5). This model is optimized for real-time, deployment-constrained environments where high throughput and low latency are critical.

---

## 📜 Citation

If you find this work or code useful, please cite:

```bibtex
@article{asandeiandradu2026efficient,
  title   = {Efficient JPEG Restoration in the Wavelet Domain via Mean Flows},
  author  = {Stefan-Alexandru Asandei and Mihai-Alexandru Radu},
  journal = {arXiv preprint arXiv:2608.28730},
  year    = {2026}
}
```
---

## 📄 License

This project is licensed under the [MIT License](./LICENSE).
