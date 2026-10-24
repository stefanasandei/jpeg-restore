# Full-image MeanFlow and trained CODiff comparison

Current paper evidence: **4,809 cases, 63 quality summaries, 14 timing configurations**. All seven configurations use the same 229 references at QF5/10/20, JPEG bitstreams, per-case seeds, metric implementations, FP32/TF32 policy and cached-JPEG-to-RGB timing boundary.

MeanFlow uses complete padded images, with NFE 1/2 and no inference tiling. FBCNN, JDEC and PromptCIR retain the fixed 224-pixel regression tile settings. CODiff uses official task-specific weights and its native VAE/latent tiling and AdaIN. See [evaluation.md](../../evaluation.md) for each operating point and GPU commands.

- `run/validation.json`: independent full-cohort, input-identity, checkpoint/backbone/source-patch, NFE, aggregate and timing validation.
- `run/per_image.jsonl`, `run/per_image.csv`: all unrounded scores, image/JPEG hashes, seeds, shapes, actual NFE and UTC measurement times.
- `run/results.json`, `run/performance.csv`, `run/quality.csv`: quality means/standard deviations, complete raw timing samples, memory, parameters and NFE.
- `run/manifest.json`: resolved settings and complete normalized source-run manifests; the original legacy manifest identifies its contributing runs.
- `export-provenance.json`: original source-file SHA-256 values and exact export transformations. Unchanged regression/JPEG cases were reused; both MeanFlow schedules and CODiff were freshly measured.
- `codiff-assets.json`: model/backbone download URLs, pinned mirror revision, hashes and actual download timestamps.
- `reused-assets-verification.json`: hashes and verification times for existing baseline/metric assets copied to the data volume, without inventing original acquisition dates.
- `codiff-source.json`, `codiff-fp32.patch`: upstream source hashes and the single FP32 cast correction.
- `figures/`: tables/quality-throughput plot generated exclusively from the validated current records.
- `SHA256SUMS`: archive file checksums.

New model downloads, isolated dependencies, runtime temporary files and original new run records are under `/mnt/projects/jpeg-restoration`; source manifests retain the exact metric-cache paths used by each run. Public exports normalize local paths and unrelated package inventory; numerical values, cryptographic identities and raw timing samples are preserved. The older tiled MeanFlow results and historical paper transcriptions do not populate this comparison. The bounded trained-checkpoint [CPU JVP audit](../matched-24-oct/jvp_public_cpu.json) is unchanged.
