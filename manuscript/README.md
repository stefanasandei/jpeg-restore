# Efficient JPEG Restoration in the Wavelet Domain via Mean Flows

The five-page paper uses the IEEE conference class, US Letter paper, standard body fonts and margins, IEEE author blocks, and numerical references. `IEEEtran.cls` and `IEEEtran.bst` retain their upstream license notices.

Run `make` with pdfLaTeX and BibTeX to produce `build/main.pdf`. The matching public PDF is `roedunet-24-oct.pdf` in the image-restoration repository. Training/evaluation code and model weights are linked through a numbered introduction footnote on page 1.

## Evidence and provenance

SODiff quality values are quoted from its paper. Ours and the five other baseline quality rows were evaluated locally, as clarified by the author. SODiff's 0.610 s timing is an SD 2.1 architectural proxy without SODiff weights; it is labeled separately in Table III and Figure 1, and no measured SODiff speedup is claimed. The original proxy script and per-repetition timing records are not archived. The current full-image PromptCIR adapter also differs from the original tiled, two-checkpoint/transpose ensemble.

Training uses all 10,000 images in LSDIR shard-00 and shard-01, alongside DF2K. Illustrative LSDIR crops are separate from quantitative benchmark aggregation; their independence from training has not been established. Architectural quality attribution and multi-seed quality evaluation remain limitations.

Table IV now reports fresh CPU FP32 agreement checks on the trained 65,323,201-parameter checkpoint: batch 1, 32x32 synthetic RGB, three seeds. Primal outputs, tangents, scalar MeanFlow-plus-severity loss, and all parameter gradients have zero observed differences. LPIPS is excluded. These cases do not establish universal or GPU equivalence. The old GPU memory-reduction/speed table is withdrawn because its allocator/backward/optimizer boundaries were not retained. A runnable audit defines those boundaries for future CUDA checks; no new CUDA run was possible in this session.

The image-restoration repository contains `evaluation.md`, `src/audit_jvp.py`, `scripts/render_paper_comparison.py`, and `artifacts/roedunet-24-oct/` with per-row provenance, CPU audit records, checkpoint/script hashes, and review disposition. Summary CSVs are transcribed historical aggregates, not original raw run logs.

## Layout

Wide quality/backbone tables and the qualitative appendix use `cuted` strips and `capt-of` captions. The backbone display starts on page 4; the appendix follows references on page 5. The loss description follows the implementation's stopped JVP correction, per-coordinate MSE, and adaptive perceptual-loss weighting.
