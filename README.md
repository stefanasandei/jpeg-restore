# Efficient JPEG Restoration in the Wavelet Domain via Mean Flows

[![arXiv](https://img.shields.io/badge/arXiv-2608.28730-b31b1b.svg)](https://arxiv.org/abs/2608.28730)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-weights-FFD21E?logo=huggingface&logoColor=white)](https://huggingface.co/asandeistefan/jpeg-restore-meanflows)

*We propose a model for generative JPEG restoration capable of up to 20 img/s, at 720p resolution on a single RTX 3090.*

## ⚡ Highlights

* **Quality–efficiency Pareto:** perception metrics close to SOTA, while being ~5x faster than SODiff
* **Wavelet-Domain Processing:** Replaces heavy learned VAEs with a **two-level 2D Haar transform**, avoiding lossy compression and latency bottlenecks.
* **Linear DiT:** a 65M-parameters model trained from scratch, with [Rank Enhanced Linear Attention](https://www.alphaxiv.org/abs/2505.16157) in every block
* **1–2 Step Inference:** Trained with **Mean Flows** for generation, without multi-stage distillation pipelines.

> Note: the current checkpoint has the best metrics for 2-steps, however similar quality can be achieved, at double the speed, with 1-step. More training, in progress, will fix this.

## 🛠️ Setup

Requirements: Python 3.12+, Pytorch 2.0+, any GPU with at least 1GB of VRAM.

Model checkpoint: [huggingface](https://huggingface.co/asandeistefan/jpeg-restore-meanflows)

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/stefanasandei/jpeg-restore.git
cd jpeg-restore
pip install -r requirements.txt
```

## 🚀 Quickstart

### 1. Inference

Run restoration on a compressed image:

```bash
python3 ./src/sample.py ./path/to/image.jpeg --checkpoint model.pt
```

### 2. Training

Train from scratch using the provided script:

```bash
python3 ./src/train.py --config-name rela_dit_meanflow
```

Check the `./config` folder for more training and eval configurations, used in other test runs and experiments.

## 📊 Benchmark Results

Check our paper for the full results. Current evaluations use 2-step inference for our model. Metrics at QF20:

| Dataset   | Method |  LPIPS ↓   |  DISTS ↓   |  MUSIQ ↑  |  MANIQA ↑  | CLIPIQA ↑  |
| --------- | ------ | :--------: | :--------: | :-------: | :--------: | :--------: |
| LIVE-1    | SODiff |   0.1237   | **0.0763** | **74.11** | **0.5272** | **0.7587** |
| LIVE-1    | *Ours* | **0.0917** |   0.1008   |   71.90   |   0.3751   |   0.7358   |
| Urban100  | SODiff |   0.0846   | **0.0734** | **72.63** | **0.5561** |   0.6733   |
| Urban100  | *Ours* | **0.0538** |   0.1368   |   70.24   |   0.4682   | **0.7172** |
| DIV2K-val | SODiff |   0.1295   | **0.0622** | **66.49** | **0.3984** |   0.6398   |
| DIV2K-val | *Ours* | **0.0995** |   0.0828   |   64.62   |   0.3427   | **0.6820** |

System-level efficiency at 1024×1024, batch size 1, single RTX 3090:

| Method | Params (M) |  NFE  | Latency (s) | Throughput (img/s) |
| ------ | :--------: | :---: | :---------: | :----------------: |
| FBCNN  |    70.1    |   1   |    0.275    |        3.63        |
| SODiff |    1288    |   1   |    0.610    |        1.64        |
| SUPIR  |    4490    |  50   |    48.66    |       0.0206       |
| *Ours* |  **65.3**  | **2** |  **0.124**  |      **8.05**      |
| *Ours* |  **65.3**  | **1** |  **0.118**  |     **17.35**      |

> *Note:* Pretrained large-scale t2i diffusion models remain stronger under extreme compression, due to their texture & patterns knowledge. This model is optimized for low latency, edge environments where high throughput is critical.

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

This project is licensed under the [MIT License](./LICENSE).
