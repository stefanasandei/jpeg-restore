# Reproducing the evaluation

Paper PDFs, manuscript sources, tests, experiment logs, and hidden directories are local-only and excluded from Git. The public release contains evaluation code, protocol documentation, and the evidence archive.

The revised comparison uses one evaluation harness and trained checkpoints for **FBCNN, JDEC, PromptCIR, and MeanFlow (one or two updates)**, plus the JPEG input. The governing preset is [`config/evaluation_matched.yaml`](config/evaluation_matched.yaml). Quality and latency use the same checkpoints, precision policy, tile grid, and reconstruction procedure. Published scores, architecture proxies, and historical ensemble timings are excluded from this comparison.

The complete run is being collected in `artifacts/matched-24-oct/run`. Until `validation.json` records a successful complete validation, partial files are progress records, not final paper results. The run comprises 4,122 image/model/QF cases and 12 timing configurations. Historical summary transcriptions under `artifacts/roedunet-24-oct` are superseded for baseline comparisons; the separate CPU JVP audit there remains valid within its documented scope.

## Run on a GPU

Use the project dependencies in `requirements.txt`, a CUDA-enabled PyTorch installation, Git, `curl`, `gdown`, the Hugging Face `hf` CLI, a C++20 compiler, and libjpeg development headers. Run from the repository root:

```bash
python scripts/prepare_matched_eval.py
bash scripts/run_matched_eval.sh outputs/matched-reproduction \
  'dataset.live1=[/path/to/LIVE1]' \
  'dataset.urban100=[/path/to/Urban100]' \
  'dataset.div2k_val=[/path/to/DIV2K_valid_HR]'
```

Preparation checks out the official source revisions, downloads the trained weights and pinned CLIP assets, installs JDEC's two extra Python dependencies in `checkpoints/matched/python-deps`, and builds its official coefficient reader locally. It checks downloaded checkpoint/CLIP bytes against the archived SHA-256 values. A rebuilt native library has a platform-dependent binary hash; its source revision and build settings identify the implementation. Do not use an unrelated checkpoint bearing the same filename.

Use a fresh output directory. Dataset paths may change: image content, dataset name, QF, and the base seed determine the sampling seed. Before comparing a reproduction, compare its image and encoded-JPEG hashes with the archived cohort. Different codec builds can produce different JPEG bytes even at the same QF.

Preparation also verifies the recorded TorchMetrics package weights and populates isolated metric caches with the archived AlexNet, VGG16, MUSIQ, MANIQA, and pinned timm ViT weights. `run_matched_eval.sh` selects those caches and sets `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`. Its preparation log distinguishes local copies from downloads. The published run uses these isolated caches. Public path placeholders identify their roles without disclosing local account paths. `metric-cache-preparation.json` records the verified local copies that populated them; the preparation and asset records distinguish those copies from fresh downloads. Do not run another GPU workload while collecting latency.

For a short functionality check, use `eval.max_images=1` and fewer timing repetitions. Such a run is deliberately rejected by the publication validator and cannot supply paper tables. The runner writes results before validation, so a failed validation retains diagnostic evidence.

## Shared input and restoration protocol

| Setting | Value |
| --- | --- |
| Evaluation images | LIVE-1: 29; Urban100: 100; DIV2K validation: 100 (`0801.png`–`0900.png`) |
| Image selection | All supported files in the configured directories/globs, sorted and deduplicated; no subset limit |
| Compression | QF 5, 10, 20; one OpenCV JPEG encode per image/QF |
| Colour and range | Pillow reference conversion to RGB; RGB→BGR for OpenCV encode; BGR→RGB after decode; float tensors in [0,1] |
| Shape preparation | Reflect-101 padding on the bottom/right to multiples of 32 **before** JPEG encoding |
| Inference grid | 224×224 tiles, nominal overlap 32, stride 192; final tile anchored to each image boundary |
| Reconstruction | Clip each tile prediction to [0,1] before equal-weight FP32 accumulation and division |
| Scoring region | Entire original reference dimensions; remove added padding only; no resize or border shaving |
| Prediction conversion | Clamp restored RGB to [0,1]; no 8-bit rounding before metrics |
| Randomness | Base seed 0; stable content-based per-image/QF seed; tile seed `(case_seed + tile_index) mod 2^32` |
| Precision | FP32 weights and tensors; TF32 enabled for both CUDA matrix multiplication and cuDNN, consistently for every model and metric |
| Execution | Batch one, eager inference, four PyTorch CPU threads, one RTX 3090 |

The 224-pixel tile size accommodates JDEC's seven-token Swin windows and the models' spatial divisibility constraints. Padding to 32 ensures that tile origins also align with JPEG blocks/chroma samples. A final boundary tile may overlap more than 32 pixels. All compared restorers use this same grid; this is a comparison of these specified operating points.

RGB models consume the decoded shared JPEG. JDEC extracts coefficients and quantization tables from **the same encoded bitstream**, then selects coefficient regions matching the common pixel tiles. Tiles are never recompressed independently. The synthesis QF is not supplied as an oracle conditioning value to any network. PromptCIR uses one released checkpoint (`model_best.pth`), without a second checkpoint, transpose ensemble, or self-ensemble. MeanFlow uses its released weights with one or two updates per tile.

The case seed is the little-endian integer represented by the first four bytes of SHA-256 over UTF-8 `base_seed/dataset/image_sha256/qf`. Tile traversal is row-major. This rule survives dataset relocation and model reordering. CUDA deterministic algorithms are not forced; recorded seeds do not imply bitwise agreement across different software/hardware stacks.

The local directory named `DF2K_valid_HR` contains precisely the 100 DIV2K validation images above; its name alone is not the split definition. The image list and hashes identify the actual evaluation cohort. Training uses all images from the first two LSDIR shards (`shard-00`, `shard-01`) alongside DF2K training. Illustrative LSDIR crops are excluded from the quantitative benchmark, and illustrative status does not establish independence from training.

## Metrics

Every restored image is scored by the same metric objects and options. The raw reports retain all seven metrics; the paper's perceptual comparison reports LPIPS, DISTS, MUSIQ, MANIQA, and CLIPIQA. Dataset/QF summaries are unweighted means over images, with population standard deviations. They are not averages of independent generation seeds or confidence intervals.

| Metric | Implementation and input convention |
| --- | --- |
| PSNR | TorchMetrics functional RGB PSNR, `data_range=1.0` |
| SSIM | TorchMetrics functional RGB SSIM, `data_range=1.0`, library defaults |
| LPIPS ↓ | TorchMetrics, AlexNet, `normalize=True` for [0,1] inputs |
| DISTS ↓ | TorchMetrics `DeepImageStructureAndTextureSimilarity` |
| MUSIQ ↑ | PyIQA `musiq`, KonIQ weights; 32-pixel patches, scales with longer sides 224/384 plus original resolution |
| MANIQA ↑ | PyIQA `maniqa`, KonIQ weights, scale 0.8; 20 uniform 224×224 evaluation crops |
| CLIPIQA ↑ | This repository's fixed CLIP ViT-B/16 prompt-pair score; pinned model, tokenizer, and image processor |

CLIPIQA averages the positive softmax probability within each pair: `Good image` / `bad image`; `Sharp image` / `blurry image`; `sharp edges` / `blurry edges`; `High resolution image` / `low resolution image`; `Noise-free image` / `noisy image`. The CLIP processor's resizing/cropping is part of this metric, applied identically to every method; restoration images themselves are not resized. This specific prompt/model implementation defines the reported score. Stateful TorchMetrics objects are reset before and after each image.

## System timing and NFE

Both 512×512 and 1024×1024 are measured at batch one on a seeded synthetic QF10 JPEG. Each model has five warm-up calls and 20 measured calls, with CUDA synchronization before and after each measured call. The timing boundary is **a cached JPEG file to a restored RGB tensor** for every method.

Included: reading the cached file, RGB decoding or JDEC coefficient extraction, host/device transfers, model preparation within the restoration path, all tile calls, overlap blending, and MeanFlow Haar transforms/sampling/decoding. Excluded: creating/staging the source JPEG, loading checkpoints, warm-up, metrics, writing predictions, and worker IPC. Filesystem caching is warm; this is not cold-storage I/O latency. Tile finite/shape checks and clipping are inside the common reconstruction path; final full-image output validation runs outside the timed interval. Performance is collected before loading metric networks.

The raw `results.json` report contains all 20 wall-clock samples, arithmetic mean, median, population standard deviation, p95 (`numpy.percentile`), and throughput `1000 / mean_ms` at batch one. GPU peak allocated/reserved memory is recorded after resetting peak statistics following warm-up; allocated peaks include resident model/input storage. No optimizer or metric model is included in the performance measurement.

NFE counts **actual network calls over all tiles**, rather than just restoration updates. There are nine tiles at 512² and 36 at 1024²: FBCNN, JDEC, PromptCIR, and one-update MeanFlow therefore use 9/36 calls; two-update MeanFlow uses 18/72. JPEG has zero network calls. The one-/two-update names describe each tile's sampling schedule.

## Asset and run provenance

[`assets.json`](artifacts/matched-24-oct/assets.json) records full SHA-256 hashes, byte sizes, source URLs, pinned Hub revisions/ETags where available, local file modification times, and acquisition evidence for checkpoints, metric weights, tokenizer/processor files, JDEC's coefficient-reader source/binary, and libjpeg. `download_verification.json` records independently repeated official downloads with explicit UTC start/completion times and matching hashes.

Dates have distinct meanings. A Hub download metadata timestamp identifies that local download completion. A local modification time may have been preserved from a server and is **not** a download timestamp. Unknown original acquisition times remain `null`; a later verification download is labeled separately. Package resources and existing caches retain their observed identities without invented acquisition dates. Download dates are provenance; the cryptographic hash identifies the bytes actually used.

Pinned baseline implementations:

| Source | Git revision |
| --- | --- |
| [FBCNN](https://github.com/jiaxi-jiang/FBCNN) | `54d1831927506b3247e2d4d245abb4f4dab1a1cd` |
| [JDEC](https://github.com/WooKyoungHan/JDEC) | `e9fe8354754dd062dbf1defebb220e290507edd2` |
| [PromptCIR](https://github.com/lbc12345/PromptCIR-NTIRE24) | `3506b1aa0e6588fdf4e0b8345d90679adf1e2220` |

MeanFlow weights: Hub revision `33315831b6b342d3b8ddbdc70aa11cb84e55c66b`; SHA-256 `fbb573706c64ae90da85e47d1ff3377734264142460770c54ebadc50a4ba4caa`. CLIP revision: `57c216476eefef5ab752ec549e440a49ae4ae5f3`. Every remaining full asset hash is in the linked manifest, rather than abbreviated in this guide.

JDEC's released checkpoint names its encoder `swinv2_rgb_nomore_deepir_rgbver`; the adapter maps that legacy name to the official `swinv2_group_embedded` factory used by its released training configuration. Encoder arguments and weights remain unchanged, with strict state-dictionary loading. Worker namespace isolation accommodates the upstream repos' module layout. The coefficient extension uses the pinned official `dct_manip.cpp`, `-O1 -std=c++20`, and `-ljpeg`; compiler and linked-library identity belong to the recorded build environment.

The public evidence export normalizes personal/workspace/dataset path prefixes and limits installed-package inventory to evaluation dependencies. Numeric results, input/checkpoint/source hashes, seeds, and raw timing samples are preserved; full original records stay locally under ignored `outputs/`. `scripts/export_matched_evidence.py` validates both copies and records the transformations and original file hashes.

The run manifest records the resolved configuration, source-file SHA-256 hashes, repository revisions, checkpoint hashes, input paths, model parameter counts, weight dtypes, worker environments, CPU thread count, TF32 flags, GPU, driver, CUDA/cuDNN, Python, and installed package versions. OpenCV and PyTorch build reports are archived separately. `system-and-upstream.json` records CPU/compiler/library details, tracked upstream checkout status, and hashes of upstream implementation files; `requirements-recorded.txt` lists the principal installed dependency versions. The measured stack is Python 3.14.4, PyTorch 2.13.0, TorchMetrics 1.8.2, PyIQA 0.1.16, Transformers 5.5.3, CUDA 13.0, and NVIDIA driver 595.71.05. Full records govern over this abbreviated list.

| File in the run directory | Contents |
| --- | --- |
| `manifest.json` | Configuration, source/environment/checkpoint identity, dataset lists |
| `per_image.jsonl`, `per_image.csv` | Every image/model/QF score, content/JPEG hashes, seed, original/encoded dimensions, actual NFE, UTC start/end times, status |
| `quality.csv` | Dataset/QF means, standard deviations, successful/expected counts |
| `performance.csv`, `results.json` | Timings, raw repetitions, memory, NFE, parameter counts, timing boundary; combined summaries |
| `errors.csv` | Explicit failures; no replacement with historical results |
| `progress.jsonl` | Append-only per-image progress while the run is active |
| `validation.json` | Successful complete-run protocol/cohort validation, generated only after checks pass |

Validate a completed run with:

```bash
python scripts/validate_matched_run.py outputs/matched-reproduction
```

The validator rejects failed/partial cohorts, subset runs, missing trained-weight identities, inconsistent precision/worker settings, different tiling, mismatched image/JPEG hashes or seeds, and incomplete timing samples. It requires the same cohort for every model. The checks also require the exact official checkpoint hashes and pinned baseline revisions, recompute quality means/standard deviations and timing means/throughput, and verify that CSV views agree with the raw JSON records. A successful run does not establish superiority beyond the tested models, datasets, QFs, and inference settings.

## Figures and independent architecture measurements

Render the quality/throughput figure directly from a validated matched run:

```bash
python src/evaluation/plots.py pareto \
  --quality outputs/matched-reproduction \
  --performance outputs/matched-reproduction \
  --dataset live1 --qf 10 --size 1024 \
  --output outputs/matched-reproduction/figures
```

The x-axis measures 1024² deployment throughput; the y-axis averages LIVE-1 QF10 scores, whose images have their original sizes. Both axes use the same model operating configuration. The plotted frontier applies only to these measured settings.

The paper's separately identified BF16, full-image, one-forward-pass backbone scaling experiment isolates architecture cost. It is not the FP32 tiled restoration comparison above. To collect a new architecture sweep and render its plots:

```bash
python src/benchmark.py eval.device=cuda eval.dtype=bfloat16 \
  eval.performance.warmup=5 eval.performance.iterations=20 \
  eval.seed=0 eval.output_dir=outputs/backbones
python src/evaluation/plots.py scaling \
  --performance outputs/backbones --output outputs/backbones/figures
```

Optional DiffBIR, SUPIR, and Rectified Flow adapters remain available in the general evaluation presets. They require their own trained weights and configurations and do not enter the matched paper comparison. There is no trained SODiff adapter/checkpoint evaluation in this repository.

## Segmented-JVP verification

The independent [`src/audit_jvp.py`](src/audit_jvp.py) audit compares the actual monolithic and segmented paths in fresh subprocesses, with identical weights, inputs/noise, and fixed `(r,t)=(0.2,0.7)`. It compares primal outputs, tangents, scalar adaptive MeanFlow-plus-severity loss, every named parameter gradient, and missing-gradient patterns. LPIPS is excluded.

The archived [`jvp_trained_cpu.json`](artifacts/roedunet-24-oct/jvp_trained_cpu.json) uses the full trained 65,323,201-parameter checkpoint, CPU FP32, batch one, 32×32 synthetic RGB, and seeds 0/1/2. All recorded differences are zero, with nonzero reference gradients for more than 59 million parameter elements in each case. Checkpoint/script hashes are included; `rtol=1e-4`, `atol=1e-6` are acceptance tolerances, not the observed errors. This verifies those cases, not every input, precision, auxiliary loss, or hardware configuration.

```bash
python src/audit_jvp.py --checkpoint /path/to/rela_dit_meanflow_qf5_20.pt \
  --device cpu --dtype float32 --crop 32 --batch 1 --seeds 0 1 2 \
  --output outputs/jvp-trained-cpu.json
```

The audit checkpoint is the full training checkpoint, rather than the inference-only export. For a fresh CUDA audit use `--device cuda --crop 128 --batch 32`; memory requirements may exceed the available GPU. Each method/seed uses a fresh process. Following its separate tangent comparison, the audit releases that graph, clears the CUDA cache, and resets peak statistics. Allocated/reserved peaks cover loss forward/backward, including resident model/input storage and parameter gradients; no optimizer/state/step is included. CPU GPU-memory fields are `null`.

The archived audit was CPU-only. The historical 15.75→6.16 GiB memory and associated speed comparison are withdrawn because their accounting boundaries were not retained. CPU gradient agreement does not validate those GPU memory values.
