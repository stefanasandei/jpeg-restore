# Matched evaluation evidence

This directory archives a fresh comparison of trained FBCNN, JDEC, PromptCIR, and MeanFlow (one/two updates), with the JPEG input as reference. No published score, historical ensemble result, or architectural timing proxy enters these results. The full protocol and reproduction commands are in [evaluation.md](../../evaluation.md).

The complete run contains six configurations × 229 images × three QFs = 4,122 per-image cases. System timings cover both 512² and 1024² at batch one, with five warm-ups and 20 measured calls. All methods share input bytes, spatial tiling/blending, FP32/TF32 policy, metrics, and a cached-JPEG-file-to-restored-RGB timing boundary.

While the GPU sweep is active, the run files are provisional progress records. A successful `run/validation.json` is required before publication. The validator checks complete image cohorts, identical image/JPEG hashes and seeds, actual tile-level NFE, model/precision settings, finite metrics, raw timing samples, and summaries recomputed from per-image records.

| Evidence | Purpose |
| --- | --- |
| `run/manifest.json` | Resolved configuration, source hashes, checkpoint identities, input lists, hardware/software and worker environments |
| `run/per_image.jsonl` | Unrounded per-image metrics, hashes, seeds, NFE, dimensions, UTC timestamps and status |
| `run/results.json` | Quality aggregates and complete timing records including every raw repetition |
| `run/*.csv` | Convenient tabular views; raw timing repetitions remain in JSON |
| `run/errors.csv` | Explicit failures; historical results never fill gaps |
| `run/validation.json` | Complete-run validation result and hashes of the validated evidence files |
| `inputs.json` | Portable image names/content hashes/dimensions, per-QF JPEG hashes and seeds; original dataset download times unknown |
| `assets.json` | Checkpoint/metric/library SHA-256 identities, source URLs and acquisition evidence |
| `download_verification.json` | Fresh official verification downloads, UTC start/end times, byte counts and matching hashes |
| `metric-hub-verification.json` | Pinned official metric Hub revisions/ETags matched against local SHA-256 hashes |
| `metric-cache-preparation.json` | Verified copies/downloads used to populate the isolated metric cache, with explicit UTC start/end times |
| `system-and-upstream.json` | CPU/compiler/library information, upstream revisions, tracked checkout status, and implementation-file hashes |
| `requirements-recorded.txt` | Principal installed dependency versions from the measured environment |
| `opencv_build.txt`, `torch_build.txt` | Codec and numerical-library build details |
| `gpu-device.csv` | GPU model, VRAM, driver, configured power limit and maximum clocks |
| `figures/` | Tables and comparison figure generated only from a validated run |

All timestamps use UTC. Modification time, original acquisition time, first observation, and verification-download time are separate fields. Google Drive can preserve a server-side modification time: it is not a local download timestamp. Unknown original dates remain null. The SHA-256 identifies bytes independently of dates and filenames. Rebuilding the native coefficient reader on another platform can change its binary hash; pinned source and build settings are also recorded.

Hashes identify this evidence; they do not imply that downloads or historical experiments occurred at an unrecorded time. The recorded package environment contains the actual measured versions, rather than a promise that every future dependency combination reproduces identical floating-point results.

Public records use portable path placeholders and retain only evaluation dependency versions. The export preserves numerical results, cryptographic identities, seeds, and all timing samples; original local records remain under ignored `outputs/`. Its transformation record identifies the normalization explicitly.

## Complete comparison and local paper build

`bash scripts/finish_paper.sh` completes the full six-configuration, three-dataset, three-QF table and the matched 512²/1024² timings. It reuses compatible per-image records and measures all missing cases. The final `run/` is exported only after all 4,122 cases and 12 timing configurations pass validation; partial subsets do not supply final paper cells. Every dataset table includes FBCNN, JDEC, PromptCIR and both MeanFlow schedules, plus the JPEG reference. No model-versus-JPEG-only results table is emitted.

Raw originals and run logs remain under ignored `outputs/matched-completion/`. Public paths are normalized and installed-package inventories are limited to evaluation dependencies. `scripts/finalize_manuscript.py` refuses incomplete evidence, empty numeric cells, missing reported values, broken resource links, unresolved references, or a manuscript exceeding five pages. The PDF and manuscript remain local-only.
