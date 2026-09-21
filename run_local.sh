#!/usr/bin/env bash
# =============================================================================
# run_local.sh -- run the ASVD baseline on a single local GPU box. No Slurm.
#
# This is the Slurm-free equivalent of the Trillium runner (baselines/run.sh in
# the ARKS comparison harness). Same compression, same evaluation, same output
# JSON -- the only thing removed is the batch scheduler.
#
# WHAT THIS MEASURES
#   ASVD performs the compression. Perplexity comes from local/shared_eval.py,
#   which transcribes the ARKS evaluation recipe, NOT from ASVD's own
#   evaluate_utils.py -- that one measures PTB on the *validation* split with
#   "\n\n".join (dense 32.55 instead of 38.99) and passes an invalid c4 config
#   name, so its numbers are not comparable to anything else.
#
# RATIO CONVENTION
#   rho = fraction of parameters KEPT (the ARKS convention).
#   --param_ratio_target = rho, passed straight through.
#   rho < 0.4 is unreachable with the upstream candidate grid (its floor is 0.4;
#   binary_search picks only from that list and settles silently at 0.4), so
#   this script sets ASVD_EXTEND_GRID=1 below 0.4. That changes the per-matrix
#   rank allocation, so a 0.33 point is NOT continuous with a 0.60 point.
#
# SINGLE-GPU / LAYERWISE -- the honest answer for this baseline
#   ASVD is whole-model-resident. There is exactly one placement decision in the
#   entire pipeline, `device_map="auto"` in local/run_asvd.py, and every stage
#   inherits it:
#     * act-aware calibration hooks EVERY nn.Linear at once and runs 256
#       whole-model forwards                             -> whole model resident
#     * the sensitivity sweep is ~98% of compression time and its metric IS
#       end-to-end calibration perplexity with one matrix swapped, so it cannot
#       be made layerwise without changing the method    -> whole model resident
#     * the binary search LOOP costs nothing here: ppl_target=-1 takes the
#       parameter-counting branch, which only sums .numel(); ~10 iterations, no
#       forward pass. But the DECOMPOSITION that follows inside the same call is
#       not free -- SVDLinear.from_linear runs torch.svd_lowrank per matrix
#       (measured 214 s over 217 matrices on opt-30b). Its working set is one
#       matrix at a time; the model around it stays resident throughout.
#   There is NO backward pass on this path. The single `.backward()` in the repo
#   is act_aware_utils.py:28 (Fisher scaling), unreachable because the driver
#   hard-codes scaling_method="abs_mean". GradSVDLinear /
#   binary_search_truncation_rank_optimize_scale have no caller at all.
#   So: one GPU big enough to hold the fp16 model is required, and that is the
#   real constraint. SHARED_EVAL_LAYERWISE=1 streams decoder layers at eval time
#   but does not help compression, which is where the memory goes.
#
# USAGE
#   ./run_local.sh setup [--force]        build the venv (transformers 4.57.6)
#   ./run_local.sh doctor                 check GPU, venv, imports, corpora
#   ./run_local.sh run MODEL RHO          one (model, rho)
#   ./run_local.sh sweep [MODEL ...]      every model x every rho, serially
#   ./run_local.sh collect                runs/*.json -> results.csv
#
#   ./run_local.sh run facebook/opt-1.3b 0.60
#   RHOS="0.60" ./run_local.sh sweep facebook/opt-125m
#
# ENVIRONMENT KNOBS (all optional)
#   VENV=<dir>        venv location                  default ./.venv-asvd
#   PYBIN=<python>    interpreter used to build it   default python3.11 else python3
#   RUNS_DIR=<dir>    result JSONs                   default ./local/runs/asvd
#   LOG_DIR=<dir>     per-run logs                   default ./local/logs
#   RHOS="0.60 0.33"  ratios for `sweep`
#   MODELS="a b c"    default model list for `sweep`
#   GPUS=0            CUDA_VISIBLE_DEVICES; "all" leaves it untouched.
#                     Defaults to a SINGLE card because every validated number
#                     was produced on one GPU and device_map="auto" would
#                     otherwise shard the model across whatever is visible.
#   OFFLINE=1         set HF_HUB_OFFLINE/HF_DATASETS_OFFLINE (caches must exist)
#   SEED=42           calibration seed
#   TORCH_SPEC        override the torch pin (see setup)
#   EXTRA_ARGS        appended verbatim to the python invocation
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

METHOD=asvd
VENV="${VENV:-$HERE/.venv-asvd}"
RUNS_DIR="${RUNS_DIR:-$HERE/local/runs/$METHOD}"
LOG_DIR="${LOG_DIR:-$HERE/local/logs}"
RHOS="${RHOS:-0.60 0.33}"
SEED="${SEED:-42}"
GPUS="${GPUS:-0}"

# Pinned to exactly what the reference runs resolved to. Both exist on PyPI.
TORCH_SPEC="${TORCH_SPEC:-torch==2.14.0}"
TRANSFORMERS_SPEC="transformers==4.57.6"
NUMPY_SPEC="numpy==2.4.2"

# The models the comparison covers, smallest first.
MODELS="${MODELS:-facebook/opt-125m facebook/opt-350m facebook/opt-1.3b \
facebook/opt-2.7b meta-llama/Llama-3.2-1B meta-llama/Llama-3.2-3B \
facebook/opt-6.7b facebook/opt-13b facebook/opt-30b meta-llama/Llama-3.1-8B}"

log()  { echo "[run_local] $*"; }
die()  { echo "[run_local] ERROR: $*" >&2; exit 1; }

# rho -> integer percent, without `bc` (absent from stock Ubuntu images).
rho_tag() { awk -v r="$1" 'BEGIN{printf "%.0f", r*100}'; }

setup_env() {
  mkdir -p "$RUNS_DIR" "$LOG_DIR"
  export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
  [[ "$GPUS" == "all" ]] || export CUDA_VISIBLE_DEVICES="$GPUS"
  local t="${LOCAL_CPUS:-$(nproc)}"
  export OMP_NUM_THREADS="$t" OPENBLAS_NUM_THREADS="$t" MKL_NUM_THREADS="$t"
  if [[ "${OFFLINE:-0}" == "1" ]]; then
    export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
  fi
}

do_setup() {
  local force=0; [[ "${1:-}" == "--force" ]] && force=1
  local py="${PYBIN:-}"
  if [[ -z "$py" ]]; then
    py=$(command -v python3.11 || command -v python3) || die "no python3 on PATH"
  fi
  if [[ -d "$VENV" && $force -eq 0 ]]; then
    log "$VENV exists -- reusing it (./run_local.sh setup --force to rebuild)"
  else
    (( force )) && rm -rf "$VENV"
    log "building $VENV with $py ($("$py" -V 2>&1))"
    "$py" -m venv "$VENV"
  fi
  "$VENV/bin/pip" install --upgrade pip setuptools wheel
  # numpy is pinned exactly: the reference environment resolved 2.4.2 here and
  # 1.26.4 in the transformers-4.45 venv from the SAME loose "numpy<3" spec.
  "$VENV/bin/pip" install "$TORCH_SPEC" "$NUMPY_SPEC" "$TRANSFORMERS_SPEC" \
      scipy scikit_learn safetensors sentencepiece datasets accelerate \
      pyarrow \
      pandas tqdm pyyaml
  "$VENV/bin/python" - <<'PY'
import torch, transformers, numpy
print(f"  transformers {transformers.__version__}  torch {torch.__version__} "
      f"(cuda {torch.version.cuda})  numpy {numpy.__version__}  "
      f"cuda_available={torch.cuda.is_available()}")
PY
}

do_doctor() {
  setup_env
  [[ -x "$VENV/bin/python" ]] || die "no venv at $VENV -- run ./run_local.sh setup"
  log "GPU:"; nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader \
    || die "nvidia-smi failed -- this script needs a CUDA GPU"
  log "python: $("$VENV/bin/python" -V 2>&1)"
  "$VENV/bin/python" - <<'PY'
import os, sys, torch, transformers
sys.path.insert(0, os.path.join(os.getcwd(), "local"))
sys.path.insert(0, os.getcwd())
print(f"  torch {torch.__version__}  cuda={torch.cuda.is_available()} "
      f"devices={torch.cuda.device_count()}")
print(f"  transformers {transformers.__version__}")
import shared_eval                      # the evaluation harness
from binary_search import binary_search_truncation_rank   # ASVD itself
from act_aware_utils import calib_input_distribution
from sensitivity import calib_sensitivity_ppl
print("  imports OK (shared_eval + ASVD)")
PY
  log "doctor passed"
}

run_one() {
  local model="$1" rho="$2"
  [[ -x "$VENV/bin/python" ]] || die "no venv at $VENV -- run ./run_local.sh setup"
  local tag="${model##*/}_rho$(rho_tag "$rho")"
  local out="$RUNS_DIR/${tag}.json"
  local logf="$LOG_DIR/${METHOD}_${tag}.log"
  mkdir -p "$RUNS_DIR" "$LOG_DIR"

  # The ONE conditional in the ASVD branch, and it is on rho, not model size.
  local grid=()
  awk -v r="$rho" 'BEGIN{exit !(r < 0.4)}' && grid=(ASVD_EXTEND_GRID=1)

  echo "=========================================================="
  echo "[run_local] method=$METHOD model=$model rho=$rho"
  echo "[run_local] out=$out"
  echo "[run_local] log=$logf"
  [[ ${#grid[@]} -gt 0 ]] && echo "[run_local] ${grid[*]}  (rho < 0.4 needs the extended candidate grid)"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
  echo "=========================================================="

  env "${grid[@]}" "$VENV/bin/python" "$HERE/local/run_asvd.py" \
      --model "$model" --rho "$rho" --alpha 0.5 \
      --n-calib-samples 32 --act-calib-samples 256 \
      --seed "$SEED" --out "$out" ${EXTRA_ARGS:-} 2>&1 | tee "$logf"
}

do_sweep() {
  setup_env
  local models=("$@"); [[ ${#models[@]} -eq 0 ]] && read -r -a models <<< "$MODELS"
  local n=0 fail=0
  for m in "${models[@]}"; do
    for r in $RHOS; do
      n=$((n+1))
      if run_one "$m" "$r"; then
        log "OK   $m rho=$r"
      else
        fail=$((fail+1)); log "FAIL $m rho=$r -- continuing"
      fi
    done
  done
  log "sweep done: $((n-fail))/$n succeeded"
  (( fail == 0 ))
}

do_collect() {
  "${VENV}/bin/python" - "$RUNS_DIR" <<'PY'
import csv, glob, json, os, re, sys
root = sys.argv[1]
# Only canonical results: <model>_rho<NN>.json. Ad-hoc debug files written into
# the same directory have previously put a 29x perplexity spread onto one
# x-axis point; the name pattern is the guard against repeating that.
CANON = re.compile(r"^[A-Za-z0-9._-]+_rho\d+\.json$")
rows, skipped = [], []
for f in sorted(glob.glob(os.path.join(root, "*.json"))):
    if not CANON.match(os.path.basename(f)):
        skipped.append(os.path.basename(f)); continue
    try:
        d = json.load(open(f))
    except Exception:
        continue
    for corpus, ppl in (d.get("ppl") or {}).items():
        rows.append({
            "method": d.get("method"), "model": d.get("model"),
            "rho": d.get("rho_target"), "dense": d.get("dense"),
            "corpus": corpus, "ppl": ppl,
            "ppl_tokens": (d.get("ppl_tokens") or {}).get(corpus),
            "realised_linear_ratio": d.get("realised_linear_ratio"),
            "realised_total_ratio": d.get("realised_total_ratio"),
            "achieved_target_ratio": d.get("achieved_target_ratio"),
            "compress_seconds": d.get("compress_seconds"), "source": f,
        })
if not rows:
    print("no results yet"); raise SystemExit(0)
rows.sort(key=lambda r: (r["model"] or "", str(r["rho"]), r["corpus"]))
out = os.path.join(root, "results.csv")
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"{len(rows)} row(s) -> {out}")
if skipped:
    print(f"skipped {len(skipped)} non-canonical file(s): {', '.join(skipped)}")
print()
print(f"{'model':<24}{'rho':>6}  {'corpus':<11}{'ppl':>12}")
for r in rows:
    print(f"{(r['model'] or '').split('/')[-1]:<24}"
          f"{(r['rho'] if r['rho'] is not None else 'dense'):>6}  "
          f"{r['corpus']:<11}{r['ppl']:>12.4f}")
PY
}

cmd="${1:-help}"; shift || true
case "$cmd" in
  setup)   do_setup "$@" ;;
  doctor)  do_doctor ;;
  run)     [[ $# -eq 2 ]] || die "usage: ./run_local.sh run MODEL RHO"
           setup_env; run_one "$1" "$2" ;;
  sweep)   do_sweep "$@" ;;
  collect) do_collect ;;
  help|-h|--help) sed -n '2,80p' "$0" | sed 's/^# \?//' ;;
  *) die "unknown command: $cmd (try ./run_local.sh help)" ;;
esac
