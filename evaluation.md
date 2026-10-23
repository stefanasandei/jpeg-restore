# Evaluation

*Quality metrics and performance benchmarks for JPEG restoration, with a shared model interface.*

## Setup

Install the project dependencies with `pip install -r requirements.txt`. Configure dataset paths in `config/evaluation.yaml`: LIVE-1, Urban100, and **DIV2K-val**. The existing `df2k_val` group is a separate, combined validation pool.

MeanFlow uses the [published checkpoint](https://huggingface.co/asandeistefan/jpeg-restore-meanflows). RF requires its own trained checkpoint. Perceptual metrics also require pretrained weights; CLIPIQA uses the existing local CLIP cache by default. Populate it with `hf download openai/clip-vit-base-patch16`, or set `eval.metric_options.clipiqa.local_files_only=false` to allow downloading.

## 1. Quality metrics

Run full-image evaluation for both MeanFlow schedules:

```bash
python3 src/eval.py \
  'eval.models=[jpeg,meanflow_1,meanflow_2]' \
  eval.checkpoints.meanflow=/path/to/model.pt \
  'dataset.live1=[/path/to/LIVE1]' \
  'eval.datasets=[live1]'
```

The default protocol evaluates each dataset separately at **QF 5, 10, and 20**, using decoded RGB in [0, 1], without resizing, tiling, border shaving, or 8-bit output rounding before scoring. JPEG inputs use the same OpenCV codec as training. JDEC receives coefficients from those exact JPEG bytes. The synthesis QF is never passed to a restoration network.

Available metrics: **PSNR, SSIM, LPIPS, DISTS, MUSIQ, MANIQA, CLIPIQA**. Scores are averaged over images, with population standard deviations. LPIPS and DISTS are lower-is-better; the others are higher-is-better. CLIPIQA retains this repo's CLIP model and prompt pairs; the model name and metric options are saved with each run.

Use `'eval.metrics=[psnr,ssim]'` for a lightweight run, `eval.max_images=10` for a smoke test, and `eval.save_images=true` to save predictions. Add `'eval.quality_factors=[1,5,10,20,50]'` to change compression levels. Seeds are fixed per image and QF, independent of model selection and iteration order. Models without trained weights are rejected by the quality pipeline.

## 2. Performance

Measure complete MeanFlow restoration, including Haar encoding, sampling, and decoding:

```bash
python3 src/eval.py \
  eval.mode=performance 'eval.models=[meanflow_1,meanflow_2]' \
  eval.checkpoints.meanflow=/path/to/model.pt eval.dtype=bfloat16 \
  'eval.performance.sizes=[[720,1280],[1024,1024]]' \
  'eval.performance.batch_sizes=[1,4]'
```

The pipeline uses seeded synthetic JPEG inputs, five warm-up calls, and 20 measured calls per shape and batch size. CUDA is synchronized around each call. Reports include mean, median, p95 and standard deviation of **batch latency**, images/s, parameter count, NFE, and peak allocated/reserved CUDA memory. Raw timings are also saved. CPU memory columns are empty.

Model loading, compilation warm-up, JPEG synthesis, metric computation, disk output, and worker communication are excluded. Model-specific coefficient parsing, conditioning, VAE encoding/decoding, and sampling remain inside the restoration call. RGB input transfers are staged before timing where the adapter supports it; DiffBIR's official host-array pipeline includes its transfers. Each report records this timing scope and the model's weight dtypes. The `jpeg` entry measures an identity operation on an already decoded input.

Set `eval.performance.warmup=10`, `eval.performance.iterations=50`, or `eval.model_specs.meanflow_2.compile=true` as needed. `eval.mode=both` runs performance before loading the quality metrics, so metric networks do not inflate measured memory.

For a controlled, eager, two-level Haar backbone sweep following the paper's protocol:

```bash
python3 src/benchmark.py
# Equivalent: python3 src/eval.py --config-name evaluation_backbones
```

This compares Restormer, SANA Linear DiT, RELA DiT, and full-attention DiT in BF16, batch size 1, at square resolutions from 64 to 1440. `adm_backbone` is also available. These entries use random initialization and report `scope=backbone`, separately from trained restoration systems. Parameter counts reflect the current local implementations. Shapes are always **[height, width]**; 720×720 differs from 720p. Both entry points use Hydra options.

## 3. Other models

Install each official implementation and its dependencies in a compatible environment. Workers need Python 3.10+, PyTorch 2+, NumPy, Pillow, OpenCV, pandas, and OmegaConf. Set `repo`, `python` (optional interpreter path), and the checkpoint/config fields under `eval.model_specs.<name>`.

| Model     | Official code                                              | Required settings                                                        |
| --------- | ---------------------------------------------------------- | ------------------------------------------------------------------------ |
| FBCNN     | [FBCNN](https://github.com/jiaxi-jiang/FBCNN)              | `repo`, `checkpoint` for the official color model                        |
| JDEC      | [JDEC](https://github.com/WooKyoungHan/JDEC)               | `repo`, `checkpoint`, compiled `dct_manip`; CUDA, batch size 1           |
| PromptCIR | [PromptCIR](https://github.com/lbc12345/PromptCIR-NTIRE24) | `repo`, `checkpoint`; single-model, full-image inference                 |
| DiffBIR   | [DiffBIR](https://github.com/XPixelGroup/DiffBIR)          | `repo`, JPEG-trained `checkpoint`, training `config`; CUDA               |
| SUPIR     | [SUPIR](https://github.com/Fanghua-Yu/SUPIR)               | `repo`, official `config` pointing to local SDXL and SUPIR weights; CUDA |

For example:

```bash
python3 src/eval.py 'eval.models=[fbcnn]' \
  eval.model_specs.fbcnn.repo=/path/to/FBCNN \
  eval.model_specs.fbcnn.checkpoint=/path/to/fbcnn_color.pth \
  eval.model_specs.fbcnn.python=/path/to/environment/bin/python
```

Each worker loads its model once and handles both pipelines. Upstream module names stay isolated. JDEC retains its JPEG-coefficient input path. PromptCIR uses one full-image forward pass, with padding removed afterward; its challenge release's tiled, two-checkpoint ensemble is a different protocol.

DiffBIR defaults to `version=custom` for the paper's **DiffBIR*** retraining. Set `version=v2.1` to evaluate the upstream generic model, which may download its own weights. SUPIR uses a fixed empty caption and full-image inference with padding; `prompt` and `sampling` options are configurable. These choices are recorded and should be matched before comparing against published tables. Upstream shape or memory failures are reported explicitly.

`evaluation_paper.yaml` selects all comparison models and the six flow configurations. Set the checkpoint and baseline paths before running it:

```bash
python3 src/eval.py --config-name evaluation_paper
```

RF entries are `rf_euler_20`, `rf_euler_50`, `rf_heun_20`, and `rf_heun_50`, sharing `eval.checkpoints.rectified_flow`. Heun uses two network evaluations per integration step. DiffBIR and SUPIR count actual denoiser calls. SODiff is excluded.

## Paper tables and plots

Run from the repository root, after setting dataset paths, `eval.checkpoints`, and the baseline settings in `config/evaluation.yaml`. The commands below each run one experiment or render its plots. Choose fresh output directories for repeat runs.

These commands remeasure the available implementations on your hardware. Published comparison rows used external results, and the current seeds, baseline settings, and local backbone implementations can differ. SODiff remains excluded.

**Full-image quality table** (`tab:quality`): LIVE-1, Urban100, and DIV2K-val, QF 5/10/20. The extra one-step MeanFlow rows also supply the Pareto plot.

```bash
python3 src/eval.py eval.mode=quality 'eval.models=[jpeg,meanflow_1,meanflow_2,fbcnn,jdec,promptcir,diffbir,supir]' 'eval.datasets=[live1,urban100,div2k_val]' 'eval.quality_factors=[5,10,20]' 'eval.metrics=[lpips,dists,musiq,maniqa,clipiqa]' eval.device=cuda eval.dtype=float32 eval.seed=0 eval.output_dir=outputs/paper/quality
```

**Flow objective and sampler table** (`tab:samplers`): both MeanFlow schedules and Euler/Heun at 20/50 integration steps, on LIVE-1.

```bash
python3 src/eval.py eval.mode=quality 'eval.models=[meanflow_1,meanflow_2,rf_euler_20,rf_euler_50,rf_heun_20,rf_heun_50]' 'eval.datasets=[live1]' 'eval.quality_factors=[5,10,20]' 'eval.metrics=[lpips,dists,musiq,maniqa,clipiqa]' eval.device=cuda eval.dtype=float32 eval.seed=0 eval.output_dir=outputs/paper/samplers
```

**System efficiency table** (`tab:system_efficiency`): complete restoration at 1024×1024, batch size 1. This also measures DiffBIR and one-step MeanFlow for the quality-throughput comparison.

```bash
python3 src/eval.py eval.mode=performance 'eval.models=[fbcnn,jdec,promptcir,diffbir,supir,meanflow_1,meanflow_2]' 'eval.performance.sizes=[[1024,1024]]' 'eval.performance.batch_sizes=[1]' eval.performance.warmup=5 eval.performance.iterations=20 eval.device=cuda eval.dtype=bfloat16 eval.seed=0 eval.output_dir=outputs/paper/systems
```

**Backbone scaling table** (`tab:backbone_raw`): the four architectures, eight square resolutions, BF16, batch size 1, without compilation.

```bash
python3 src/benchmark.py eval.device=cuda eval.dtype=bfloat16 eval.performance.warmup=5 eval.performance.iterations=20 eval.seed=0 eval.output_dir=outputs/paper/backbones
```

The quality and sampler tables are in each run's `quality.csv`. System and backbone tables are in `performance.csv`; latency is recorded in milliseconds, so divide by 1000 for the paper's seconds column. Compilation remains disabled in these presets.

**Throughput and memory scaling plots** (`fig:backbone_scaling`): render both plots from the backbone sweep.

```bash
python3 src/evaluation/plots.py scaling --performance outputs/paper/backbones --output outputs/paper/figures
```

This writes `throughput_vs_resolution.{png,pdf}` and `memory_vs_resolution.{png,pdf}`. Memory plotting requires CUDA readings.

**Quality-throughput Pareto plot** (`fig:pareto`): combine LIVE-1 QF10 LPIPS with batch-one 1024×1024 system throughput. The plot uses logarithmic throughput and marks the measured frontier. Quality uses FP32 and timing uses BF16, as specified above; use matching precision in both collection commands when comparing a single precision setting.

```bash
python3 src/evaluation/plots.py pareto --quality outputs/paper/quality --performance outputs/paper/systems --dataset live1 --qf 10 --size 1024 --output outputs/paper/figures
```

This writes `pareto_quality_vs_throughput.{png,pdf}` and `pareto_points.csv`. Plotting rejects failed or partial reports and pairs models by name, excluding the JPEG identity benchmark.

**Appendix predictions:** prepare the fixed 512×512 LSDIR crops first, then save JPEG, one-step, and two-step outputs at QF 5/10. Reference images remain in the input directory; panel selection and layout are manual.

```bash
python3 src/eval.py eval.mode=quality 'eval.models=[jpeg,meanflow_1,meanflow_2]' '+dataset.lsdir_figures=[/path/to/fixed_512_crops]' 'eval.datasets=[lsdir_figures]' 'eval.quality_factors=[5,10]' 'eval.metrics=[psnr,ssim]' eval.save_images=true eval.device=cuda eval.seed=0 eval.output_dir=outputs/paper/qualitative
```

## Results and extensions

Each run writes to `outputs/evaluation/<timestamp>/`: `quality.csv`, `per_image.csv`, `per_image.jsonl`, `performance.csv`, `results.json`, `errors.csv`, and `manifest.json`. The manifest records resolved settings, input paths, seeds, package versions, hardware, Git revisions, checkpoint hashes, and source fingerprints. Use `eval.output_dir=/path/to/run` for a fixed destination; existing results require `eval.overwrite=true`. `eval.hash_weights=false` disables file hashing.

Failures remain in the reports and produce a nonzero exit status after the selected models finish. Partial quality rows include successful and expected image counts. Set `eval.continue_on_error=false` to stop at the first failure. Historical paper values are never substituted for measurements.

Add metrics through `src/metrics/suite.py` and the `ImageMetric` interface. Add model presets in `config/evaluation.yaml`; new integrations implement `Adapter.prepare`, `restore`, and `parameter_count`. A custom preset can use `kind: custom` with `factory: package.module.callable`, returning an adapter from `(spec, device, dtype)`. The runner, timing code, and reports are shared in `src/evaluation/`.
