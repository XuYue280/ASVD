#!/usr/bin/env python3
"""Compress with ASVD, then evaluate through the shared (ARKS-equivalent) harness.

ASVD's own evaluate_utils.py measures PTB on the *validation* split with
"\\n\\n".join (dense 32.55, not 38.99) and passes an invalid c4 config name, so
its numbers are not comparable to ARKS. Here ASVD only does the compression;
perplexity comes from tools/shared_eval.py, the same code every baseline uses.
"""
# VENDORED COPY -- the repo root is resolved from this file's location
# instead of the hard-coded /scratch path the Trillium harness used, so
# this runs unchanged from any checkout directory.
import argparse, json, os, random, sys, time
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)          # <checkout>/local/..  ->  <checkout>
sys.path.insert(0, _REPO)
sys.path.insert(0, _HERE)

from shared_eval import evaluate_all, count_parameters


class _Shield(torch.nn.Module):
    """Wrap a Linear so ASVD's isinstance(nn.Linear) walk skips it."""
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
    def forward(self, x):
        return self.inner(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rho", type=float, required=True, help="fraction of params KEPT")
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib-dataset", default="wikitext2")
    ap.add_argument("--n-calib-samples", type=int, default=32,
                    help="samples for the SENSITIVITY sweep")
    ap.add_argument("--act-calib-samples", type=int, default=256,
                    help="samples for the ACT-AWARE scaling (separate set)")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--sensitivity-metric", default="ppl", choices=["ppl","stable_rank"])
    ap.add_argument("--skip-lm-head", action="store_true",
                    help="exclude lm_head from compression (most SVD papers do)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dense", action="store_true", help="skip compression (baseline run)")
    a = ap.parse_args()

    # ASVD calls random.randint in datautils but never seeds `random`; without
    # this the calibration sample choice is not reproducible run to run.
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, use_fast=False, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        a.model, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    )
    before = count_parameters(model)
    t0 = time.perf_counter()

    if not a.dense:
        from datautils import get_calib_data
        from act_aware_utils import calib_input_distribution
        from sensitivity import calib_sensitivity_ppl, calib_sensitivity_stable_rank
        from binary_search import binary_search_truncation_rank

        args = argparse.Namespace(
            model_id=a.model, ppl_target=-1, param_ratio_target=a.rho,
            act_aware=True, alpha=a.alpha, n_calib_samples=a.n_calib_samples,
            calib_dataset=a.calib_dataset, scaling_method="abs_mean",
            sensitivity_metric=a.sensitivity_metric, use_cache=False, weight_quant="none",
            eval_mmlu=False, sigma_fuse="UV", seed=a.seed, DEV="cuda",
            compress_kv_cache=False, kv_cache_ratio_target=-1, eval_tasks="",
            rank_align=1,
        )
        if a.skip_lm_head:
            # ASVD's module walk (binary_search.py:15-27) picks up every nn.Linear
            # including lm_head, which is 31% of opt-125m's counted params.
            import torch.nn as nn
            head = getattr(model, "lm_head", None)
            if isinstance(head, nn.Linear):
                model.lm_head = _Shield(head)

        # Two distinct calibration sets. Upstream asvd.py reuses one loader for
        # both, but the act-aware scaling wants far more data than the
        # sensitivity sweep (256 vs 32); sharing the 32-sample set under-
        # estimates the activation statistics that weight the SVD.
        # get_calib_data's `seed` argument only names the cache file -- the
        # sampling itself pulls from the global `random` stream. Re-seeding
        # before each set makes both draw "the first N" windows independently,
        # rather than the sensitivity set continuing where the act-aware set
        # stopped. ASVD_SHARED_STREAM=1 keeps the sequential behaviour.
        act_calib = get_calib_data(a.calib_dataset, tok, a.model,
                                   a.act_calib_samples, seed=a.seed)
        calib_input_distribution(model, act_calib, args.scaling_method, use_cache=False)
        if os.environ.get("ASVD_SHARED_STREAM", "") != "1":
            random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
        calib = get_calib_data(a.calib_dataset, tok, a.model, a.n_calib_samples, seed=a.seed)
        # binary_search_truncation_rank iterates sensitivity_dict (binary_search.py:34)
        # and cannot take None -- the per-layer sensitivity sweep has to run first.
        _sens_fn = (calib_sensitivity_stable_rank
                    if a.sensitivity_metric == "stable_rank" else calib_sensitivity_ppl)
        sensitivity = _sens_fn(model, calib, args, use_cache=False)
        binary_search_truncation_rank(model, sensitivity, calib, args)

    compress_s = time.perf_counter() - t0
    after = count_parameters(model)

    # Recipe cross-checks: the whole-model achieved ratio over the 72 target
    # matrices, and how many of them the binary search left fully dense.
    import torch.nn as nn
    tgt = ("q_proj", "k_proj", "v_proj", "out_proj", "o_proj", "fc1", "fc2",
           "gate_proj", "up_proj", "down_proj")
    dense_kept, n_target, comp_p, dense_p = 0, 0, 0, 0
    for name, mod in model.named_modules():
        if not any(name.endswith(t) for t in tgt):
            continue
        n_target += 1
        if isinstance(mod, nn.Linear):
            dense_kept += 1
            comp_p += mod.weight.numel(); dense_p += mod.weight.numel()
        else:
            p = sum(q.numel() for q in mod.parameters() if q.dim() == 2)
            comp_p += p
            ranks = [q for q in mod.parameters() if q.dim() == 2]
            if len(ranks) >= 2:
                dense_p += ranks[0].shape[0] * ranks[1].shape[1]
    achieved = comp_p / dense_p if dense_p else None
    print(f"[check] target matrices={n_target}  kept fully dense={dense_kept}  "
          f"achieved ratio={achieved:.4f}" if achieved else "[check] n/a", flush=True)

    res = evaluate_all(model, tok, device="cuda")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    payload = {
        "method": "asvd", "model": a.model,
        "rho_target": None if a.dense else a.rho,
        "dense": a.dense, "seed": a.seed,
        "calib_dataset": a.calib_dataset, "n_calib_samples": a.n_calib_samples,
        "act_calib_samples": a.act_calib_samples,
        "params_before": before, "params_after": after,
        "target_matrices": n_target, "kept_dense": dense_kept,
        "achieved_target_ratio": achieved,
        "realised_total_ratio": after["total_params"] / before["total_params"],
        "realised_linear_ratio": after["linear_params"] / before["linear_params"],
        "compress_seconds": compress_s,
        "ppl": {c: v["ppl"] for c, v in res.items()},
        "ppl_tokens": {c: v["ppl_tokens"] for c, v in res.items()},
    }
    with open(a.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print("\n" + json.dumps(payload["ppl"], indent=2))
    print(f"realised linear ratio = {payload['realised_linear_ratio']:.4f}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
