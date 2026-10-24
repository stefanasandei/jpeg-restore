# Historical summaries and separate CPU JVP evidence

**Baseline comparisons in this directory are superseded by the fresh [matched evaluation](../matched-24-oct). They are historical records and do not supply the current comparison tables or figure.**

Manuscript sources and paper PDFs are local-only and excluded from Git. Full protocols, supported adapters, limitations, and commands are in [`../../evaluation.md`](../../evaluation.md).

## Original summaries

`quality_table.csv` transcribes Table I, including per-row provenance. Per the author's clarification, all methods except SODiff were evaluated locally; SODiff quality scores come from its paper. `system_table.csv` transcribes the reported system summary, with the SD 2.1 architectural proxy labeled separately. These files preserve reported aggregate values; they are not original raw evaluation records, independent reruns, or evidence of checkpoint-level SODiff timing.

The previous Figure 1 was generated from these historical CSVs. The current `scripts/render_paper_comparison.py` requires a complete validated matched run and does not accept these transcribed summaries. Its hollow SODiff-quality/proxy-time point is illustrative. No frontier or SODiff speedup is established by that pairing. The original one-update point is omitted because this figure uses the operating points tabulated in the revised paper.

## New JVP verification

`jvp_trained_cpu.json` records three fresh CPU FP32 cases on the full trained 65,323,201-parameter checkpoint at batch 1 and 32×32 input size. `jvp_smoke_cpu.json` records a separate small architecture check. Both use the actual training implementation. All primal, tangent, loss, and parameter-gradient comparisons passed with zero observed error. The trained cases have more than 59 million nonzero reference gradient elements each, so the result is not an all-zero gradient comparison.

The reports contain the checkpoint SHA-256, audit script SHA-256, settings, environment, metrics, and loss/memory scope. Paths are published as basenames. `src/audit_jvp.py` regenerates the checks; checkpoint bytes must match the recorded hash to reproduce the same trained case. The synthetic inputs isolate derivative behavior and are not restoration-quality evaluations.

These archived derivative checks were performed on CPU. No new GPU memory, throughput, or 128×128/batch-32 verification was performed. The historical JVP GPU memory/speed table is withdrawn from the paper because its measurement boundaries cannot be recovered. New CPU verification does not repair those historical GPU records.

## Review disposition

| Concern | Revision | Remaining evidential limit |
| --- | --- | --- |
| Public documentation conflicts with paper | Corrected paper, README, evaluation guide, and CSV provenance together | Original baseline configurations/raw logs are not all archived |
| SODiff speed claims | Removed measured SODiff speedup; labeled SD 2.1 proxy in table, figure, abstract, and documentation | Original proxy script/configuration is unavailable; no SODiff checkpoint was evaluated |
| Tangent/loss/parameter-gradient equivalence | Added runnable audit and three trained-checkpoint CPU results | Finite FP32 CPU cases; LPIPS auxiliary loss excluded; no universal/GPU equivalence claim |
| Undefined JVP peak memory | Withdrew historical memory-reduction claim; new CUDA audit specifies allocated/reserved, baseline, backward, and optimizer exclusions | CUDA audit has not been run |
| Training subset | Names shard-00 and shard-01, used in full | Appendix examples are illustrative; their independence from training has not been established |
| Quality contribution of individual components | Explicit limitation; backbone cost does not establish quality attribution | No architectural quality ablation or multi-seed quality study was run |

These revisions address reporting and release consistency. They do not constitute a claim of reviewer acceptance or a guaranteed review score.
