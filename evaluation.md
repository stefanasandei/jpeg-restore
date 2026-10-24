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

Input selection uses every supported image file in the configured directories/globs, sorted and deduplicated; `eval.max_images` explicitly limits that list. Training uses the directories listed for both complete LSDIR shards in `config/rf_pretrain.yaml`. Evaluation image paths are recorded in each manifest, and per-image reports retain seeds and failures. The default DIV2K-val path is a placeholder and must point to the actual DIV2K validation split, not the combined DF2K validation directory.

The current quality harness derives each seed from the configured base seed, dataset name, **absolute image path**, and QF. It is stable across model ordering on the same paths; relocating datasets changes seeds. Thus `eval.seed=0` specifies a base seed and does not mean every image receives literal seed zero, nor does it guarantee regeneration of historical predictions. Retain the original path mapping and per-image seed records when auditing a run.

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

`evaluation_paper.yaml` selects the currently supported comparison adapters and six flow configurations. It does not reproduce every historical paper operating point. Set the checkpoint and baseline paths before running it:

```bash
python3 src/eval.py --config-name evaluation_paper
```

RF entries are `rf_euler_20`, `rf_euler_50`, `rf_heun_20`, and `rf_heun_50`, sharing `eval.checkpoints.rectified_flow`. Heun uses two network evaluations per integration step. DiffBIR and SUPIR count actual denoiser calls. SODiff is not implemented: its paper quality scores are external references, and the original SD 2.1 architectural timing proxy is not a SODiff checkpoint evaluation.

## Paper tables and plots

Run from the repository root, after setting dataset paths, `eval.checkpoints`, and the baseline settings in `config/evaluation.yaml`. The commands below each run one experiment or render its plots. Choose fresh output directories for repeat runs.

The paper combines original local measurements, explicitly attributed external quality scores, and a separately labeled architectural timing proxy. The CSVs in [`artifacts/roedunet-24-oct`](artifacts/roedunet-24-oct) are transcriptions of the reported summaries, **not raw logs or fresh measurements**.

| Paper result | Origin | What the current commands reproduce |
| --- | --- | --- |
| Ours, FBCNN, JDEC, PromptCIR, DiffBIR*, SUPIR quality | Original local evaluations, as confirmed by the author | New measurements with the specified checkpoints and current adapters; historical checkpoints/configurations are not all archived |
| SODiff quality | Published SODiff paper, https://arxiv.org/abs/2508.07346 | No local SODiff quality evaluation; no SODiff checkpoint was evaluated |
| Ours, FBCNN, JDEC, SUPIR system timings | Original local RTX 3090 measurements | Fresh timings with current adapters; historical raw repetitions are not included |
| PromptCIR system timing | Original local tiled, two-checkpoint/transpose ensemble | The current adapter measures a different full-image, single-checkpoint path; it does not reproduce the historical 64-call row |
| SD 2.1 proxy timing | Local SD 2.1 reconstruction following the SODiff architecture, without SODiff weights | Original proxy script/configuration and raw repetitions are not archived; this is not validated SODiff latency |
| Backbone scaling | Original local forward-only measurements | A new sweep of current local implementations, whose parameter counts can differ from the historical models |
| Table IV: JVP agreement | New CPU FP32 audit of the trained checkpoint, three seeds, batch 1, 32×32 synthetic RGB | Runnable with `src/audit_jvp.py`; reports and checkpoint hash are archived below |

Accordingly, the paper does not claim a measured SODiff speedup, a fully matched quality-throughput frontier, or exact reproduction of all historical tables from the current presets. The first two LSDIR shards (`shard-00`, `shard-01`) are used in full for the 10,000-image training subset, alongside DF2K. The qualitative LSDIR examples are illustrative and are not part of LIVE-1, Urban100, or DIV2K-val quality aggregation; their independence from training is not established by their illustrative status.

The following commands generate **new run reports**. Do not substitute historical table cells for unavailable measurements, and do not present a new adapter's result as the historical operating point.

**Quality measurements corresponding to Table I** (`tab:quality`): LIVE-1, Urban100, and DIV2K-val, QF 5/10/20. This command does not generate the externally sourced SODiff row or guarantee the historical baseline configurations.

```bash
python3 src/eval.py eval.mode=quality 'eval.models=[jpeg,meanflow_1,meanflow_2,fbcnn,jdec,promptcir,diffbir,supir]' 'eval.datasets=[live1,urban100,div2k_val]' 'eval.quality_factors=[5,10,20]' 'eval.metrics=[lpips,dists,musiq,maniqa,clipiqa]' eval.device=cuda eval.dtype=float32 eval.seed=0 eval.output_dir=outputs/paper/quality
```

**Optional sampler extension** (not a table in the current five-page paper): both MeanFlow schedules and Euler/Heun at 20/50 integration steps, on LIVE-1.

```bash
python3 src/eval.py eval.mode=quality 'eval.models=[meanflow_1,meanflow_2,rf_euler_20,rf_euler_50,rf_heun_20,rf_heun_50]' 'eval.datasets=[live1]' 'eval.quality_factors=[5,10,20]' 'eval.metrics=[lpips,dists,musiq,maniqa,clipiqa]' eval.device=cuda eval.dtype=float32 eval.seed=0 eval.output_dir=outputs/paper/samplers
```

**New system measurements corresponding to Table III** (`tab:system_efficiency`): complete restoration at 1024×1024, batch size 1. The full-image PromptCIR result differs from the historical ensemble; neither SODiff nor the original SD 2.1 proxy is run. This command additionally measures DiffBIR and one-step MeanFlow.

```bash
python3 src/eval.py eval.mode=performance 'eval.models=[fbcnn,jdec,promptcir,diffbir,supir,meanflow_1,meanflow_2]' 'eval.performance.sizes=[[1024,1024]]' 'eval.performance.batch_sizes=[1]' eval.performance.warmup=5 eval.performance.iterations=20 eval.device=cuda eval.dtype=bfloat16 eval.seed=0 eval.output_dir=outputs/paper/systems
```

**Backbone scaling table** (`tab:backbone_raw`): the four architectures, eight square resolutions, BF16, batch size 1, without compilation.

```bash
python3 src/benchmark.py eval.device=cuda eval.dtype=bfloat16 eval.performance.warmup=5 eval.performance.iterations=20 eval.seed=0 eval.output_dir=outputs/paper/backbones
```

New quality and optional sampler results are in each run's `quality.csv`. System and backbone results are in `performance.csv`; latency is recorded in milliseconds, so divide by 1000 for the paper's seconds column. Compilation remains disabled in these presets.

**Throughput and memory scaling plots** (`fig:backbone_scaling`): render both plots from the backbone sweep.

```bash
python3 src/evaluation/plots.py scaling --performance outputs/paper/backbones --output outputs/paper/figures
```

This writes `throughput_vs_resolution.{png,pdf}` and `memory_vs_resolution.{png,pdf}`. Memory plotting requires CUDA readings.

**Comparison plot from new runs:** combine LIVE-1 QF10 LPIPS with batch-one 1024×1024 system throughput. The plot marks non-dominated points among the paired reports, not a universal frontier. These example commands collect quality in FP32 and timing in BF16; they are distinct operating settings, not demonstrated equal-quality BF16 deployment. Set matching precision when collecting quality and speed for a single operating point. SODiff is absent.

```bash
python3 src/evaluation/plots.py pareto --quality outputs/paper/quality --performance outputs/paper/systems --dataset live1 --qf 10 --size 1024 --output outputs/paper/figures
```

This writes `pareto_quality_vs_throughput.{png,pdf}` and `pareto_points.csv`. Plotting rejects failed or partial reports and pairs models by name, excluding the JPEG identity benchmark.

**Historical Figure 1:** render the separately archived summary cells with explicit provenance:

```bash
python3 scripts/render_paper_comparison.py
```

The hollow point combines SODiff's published LPIPS with the SD 2.1 proxy timing. It is illustrative and is not a checkpoint-level SODiff measurement. The renderer does not evaluate a model or generate new evidence.

**Appendix predictions:** prepare the fixed 512×512 LSDIR crops first, then save JPEG, one-step, and two-step outputs at QF 5/10. Reference images remain in the input directory; panel selection and layout are manual.

```bash
python3 src/eval.py eval.mode=quality 'eval.models=[jpeg,meanflow_1,meanflow_2]' '+dataset.lsdir_figures=[/path/to/fixed_512_crops]' 'eval.datasets=[lsdir_figures]' 'eval.quality_factors=[5,10]' 'eval.metrics=[psnr,ssim]' eval.save_images=true eval.device=cuda eval.seed=0 eval.output_dir=outputs/paper/qualitative
```

## Segmented-JVP verification (Table IV)

The audit runs the actual `MeanFlow` monolithic (`jvp`) and segmented (`rcm`) paths in fresh subprocesses, with identical model weights, synthetic inputs, noise, and fixed times `r=0.2`, `t=0.7`. It compares primal outputs, derivative tangents, scalar loss, every named parameter gradient, and the pattern of parameters without gradients. The loss includes adaptive MeanFlow and severity regression; **LPIPS is excluded**. Gradient agreement here does not establish agreement for an auxiliary LPIPS loss or every input/precision/hardware combination.

The archived [`jvp_trained_cpu.json`](artifacts/roedunet-24-oct/jvp_trained_cpu.json) uses the full 65,323,201-parameter trained checkpoint, CPU FP32, batch 1, 32×32 synthetic RGB, and seeds 0/1/2. All recorded errors are zero, with nonzero reference gradients for more than 59 million parameter elements in each case. Checkpoint and audit-script SHA-256 hashes are recorded. Comparison tolerances are `rtol=1e-4`, `atol=1e-6`; zero errors are the observed values, not merely a pass at those tolerances. Local paths are replaced by basenames in the published reports.

```bash
python3 src/audit_jvp.py \
  --checkpoint /path/to/rela_dit_meanflow_qf5_20.pt \
  --device cpu --dtype float32 --crop 32 --batch 1 --seeds 0 1 2 \
  --output outputs/jvp-trained-cpu.json
```

A small nonzero-head architecture check is also archived as [`jvp_smoke_cpu.json`](artifacts/roedunet-24-oct/jvp_smoke_cpu.json). It tests the code paths and is not evidence about trained-model quality:

```bash
python3 src/audit_jvp.py --smoke --crop 16 --batch 1 --seeds 0 1 2
```

For a fresh CUDA verification at the original training geometry:

```bash
python3 src/audit_jvp.py \
  --checkpoint /path/to/rela_dit_meanflow_qf5_20.pt \
  --device cuda --dtype float32 --crop 128 --batch 32 --seeds 0 1 2 \
  --output outputs/jvp-trained-cuda.json
```

**Memory definition:** each method/seed uses a fresh process. Following the separate tangent comparison, the audit releases its graph, clears the CUDA cache, and resets peak statistics. Reported peaks are total allocated and reserved CUDA memory during the loss forward and backward; the allocated baseline includes resident model/input storage. Parameter-gradient storage is included. No optimizer is constructed, no optimizer state is allocated, and no optimizer step is timed. CPU runs report `null` for GPU memory. The single synchronized forward/backward duration is diagnostic, not a warmed throughput benchmark.

CUDA was unavailable for the archived audit. The paper therefore withdraws the historical 15.75→6.16 GiB memory-reduction and associated speed comparison: their allocator/backward/optimizer boundaries were not retained. The new CPU agreement results do not validate those GPU peaks. CUDA OOM or unavailable-device errors must not be replaced by CPU or historical numbers.

## Results and extensions

Each run writes to `outputs/evaluation/<timestamp>/`: `quality.csv`, `per_image.csv`, `per_image.jsonl`, `performance.csv`, `results.json`, `errors.csv`, and `manifest.json`. The manifest records resolved settings, input paths, seeds, package versions, hardware, Git revisions, checkpoint hashes, and source fingerprints. Use `eval.output_dir=/path/to/run` for a fixed destination; existing results require `eval.overwrite=true`. `eval.hash_weights=false` disables file hashing.

Failures remain in the reports and produce a nonzero exit status after the selected models finish. Partial quality rows include successful and expected image counts. Set `eval.continue_on_error=false` to stop at the first failure. Historical paper values are never substituted for measurements.

Add metrics through `src/metrics/suite.py` and the `ImageMetric` interface. Add model presets in `config/evaluation.yaml`; new integrations implement `Adapter.prepare`, `restore`, and `parameter_count`. A custom preset can use `kind: custom` with `factory: package.module.callable`, returning an adapter from `(spec, device, dtype)`. The runner, timing code, and reports are shared in `src/evaluation/`.
