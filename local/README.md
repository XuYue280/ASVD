# `local/` — single-GPU runner support files

Everything here exists so `../run_local.sh` can run this baseline on an ordinary
GPU box (e.g. Lambda Labs) with no Slurm and no cluster-specific paths.

| file | origin | change |
|---|---|---|
| `run_asvd.py` | ARKS comparison harness, `baselines/tools/run_asvd.py` | repo root resolved from `__file__` instead of a hard-coded `/scratch` path |
| `shared_eval.py` | ARKS comparison harness, `baselines/tools/shared_eval.py` | `LOCAL_FILES` made environment-driven (`SHARED_EVAL_PTB_FILE`, `SHARED_EVAL_C4_FILE`) instead of hard-coded cluster paths |

Nothing else was touched. The compression code in the repo root is upstream ASVD
with the patches this project needed; the evaluation is `shared_eval.py`, which
transcribes the ARKS recipe so the number is comparable across baselines.

## Single-GPU / layerwise / backward — the short answer

* **Backward pass: none.** The only `.backward()` in the repo is
  `act_aware_utils.py` (Fisher scaling) and it is unreachable — the driver
  hard-codes `scaling_method="abs_mean"`. `GradSVDLinear` and
  `binary_search_truncation_rank_optimize_scale` have no caller at all.
* **Single GPU: yes**, for every model that fits. Every reference result,
  including opt-30b (59.9 GB fp16), was produced on one 80 GB card.
* **Layerwise: no**, and the expensive stage cannot be made layerwise. The
  sensitivity sweep is ~98% of compression time and its metric *is* end-to-end
  calibration perplexity with one matrix swapped, so every evaluation has to
  traverse the full stack. `SHARED_EVAL_LAYERWISE=1` streams decoder layers at
  *evaluation* time only; it does not help compression, which is where the
  memory goes.

The practical constraint is therefore: **one GPU large enough to hold the fp16
model.** See the header of `../run_local.sh` for the per-stage detail.
