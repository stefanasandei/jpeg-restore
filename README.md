# Efficient JPEG Restoration in the Wavelet Domain via Mean Flows

[![arXiv](https://img.shields.io/badge/arXiv-2608.28730-b31b1b.svg)](https://arxiv.org/abs/2608.28730)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-weights-FFD21E?logo=huggingface&logoColor=white)](https://huggingface.co/asandeistefan/jpeg-restore-meanflows)

*A 65M-parameter generative JPEG restorer trained from scratch for one- or two-update MeanFlow sampling.*

## ⚡ Highlights

* **Auditable evaluation:** shared JPEG inputs, trained checkpoints, precision, metrics, and timing boundaries across all compared methods.
* **Wavelet-Domain Processing:** Replaces heavy learned VAEs with a **two-level 2D Haar transform**, avoiding lossy compression and latency bottlenecks.
* **Linear DiT:** a 65M-parameters model trained from scratch, with [Rank Enhanced Linear Attention](https://www.alphaxiv.org/abs/2505.16157) in every block
* **1–2 Step Inference:** Trained with **Mean Flows** for generation, without multi-stage distillation pipelines.

## 🛠️ Setup

Install the project dependencies below. The recorded matched evaluation uses an RTX 3090 with 24 GiB VRAM; its exact software environment is archived with the results.

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

### 3. Evaluation

See [evaluation.md](./evaluation.md) for the matched quality/performance protocol, asset hashes, acquisition metadata, and GPU reproduction command.

## 📊 Benchmark Results

<!-- MATCHED_RESULTS_START -->
The complete comparison validates **4,809 image cases**, **63 quality summaries**, and **14 timing configurations** on LIVE-1, Urban100 and DIV2K validation at QF5/10/20. FBCNN, JDEC, PromptCIR, CODiff and both MeanFlow schedules share JPEG inputs, metrics, FP32/TF32 precision and timing boundaries. **MeanFlow processes whole images without tiling**, in one or two evaluations; baseline inference settings are documented separately.

At LIVE-1 QF10, two-update MeanFlow obtains LPIPS **0.1596**, versus **0.1427** for CODiff. Its matched 1024² throughput is **3.090 images/s**, versus **0.565** for CODiff. These are measured quality/compute operating points.

See [complete validated records](artifacts/full-image-codiff/run) and [reproduction protocol](evaluation.md). Paper PDFs and manuscript sources remain local.
<!-- MATCHED_RESULTS_END -->

The independent [JVP audit](src/audit_jvp.py) and [CPU records](artifacts/matched-24-oct/jvp_public_cpu.json) compare outputs, tangents, losses, and parameter gradients on the trained checkpoint. These bounded CPU checks do not establish historical GPU memory reductions.

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
