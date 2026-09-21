import os
import torch
import torch.nn as nn
from evaluate_utils import evaluate_model, evaluate_perplexity
from modules.svd_linear import SVDLinear, GradSVDLinear
from tqdm import tqdm
import time



def _asvd_skip(full_name):
    """Exclude the output head from compression.

    ASVD's module walk picks up every nn.Linear, lm_head included. On opt-125m
    lm_head is 38.6M of the 123.5M counted parameters (31%), so a nominal
    param_ratio of 0.6 spends most of its budget on the head and the decoder is
    cut far harder than the number suggests. ARKS and Basis_Sharing both
    compress decoder linears only, so leaving the head alone is what makes the
    three comparable -- and it reproduces the published ASVD numbers.
    Set ASVD_COMPRESS_LM_HEAD=1 to restore the original behaviour.
    """
    import os
    if os.environ.get("ASVD_COMPRESS_LM_HEAD", "") == "1":
        return False
    return "lm_head" in full_name or "embed_" in full_name


def binary_search_truncation_rank(model, sensitivity_dict, calib_loader, args):
    module_dict = {name: module for name, module in model.named_modules()}
    full_name_dict = {module: name for name, module in model.named_modules()}
    linear_info = {}
    modules = [model]
    while len(modules) > 0:
        submodule = modules.pop()
        for name, raw_linear in submodule.named_children():
            if isinstance(raw_linear, nn.Linear):
                full_name = full_name_dict[raw_linear]
                if _asvd_skip(full_name):
                    continue

                linear_info[raw_linear] = {
                    "father": submodule,
                    "name": name,
                    "full_name": full_name,
                }
            else:
                modules.append(raw_linear)

    if args.compress_kv_cache:
        ratio_target = args.kv_cache_ratio_target
        sensitivity_dict = {k: v for k, v in sensitivity_dict.items() if "k_proj" in k or "v_proj" in k}
        assert args.ppl_target < 0, "ppl_target is not supported when compressing kv_cache"
        default_param_ratio = 2
    else:
        ratio_target = args.param_ratio_target
        default_param_ratio = 1

    print(
        f"=== {'compress kv_cache' if args.compress_kv_cache else 'compress weight'} target: ppl={args.ppl_target}, ratio_target={ratio_target} ==="
    )

    sensitivity_list = []
    for layername, v in sensitivity_dict.items():
        for param_ratio, ppl in v.items():
            if not args.compress_kv_cache and param_ratio >= 1:
                # we need to compress the weights, so parameter ratio should be less than 1
                continue
            sensitivity_list.append((layername, param_ratio, ppl))
    sorted_sensitive_list = sorted(sensitivity_list, key=lambda x: -x[2])

    # binary search
    high = len(sorted_sensitive_list) - 1
    low = 0
    assert args.ppl_target > 0 or ratio_target > 0

    input_ids = torch.cat([_["input_ids"] for _ in calib_loader], 0)
    while low < high:
        mid = (low + high) // 2
        layers_min_ratio = {layername: default_param_ratio for layername in sensitivity_dict.keys()}
        for layername, param_ratio, ppl in sorted_sensitive_list[mid:]:
            layers_min_ratio[layername] = min(layers_min_ratio[layername], param_ratio)
        tot_params = 0
        compress_params = 0
        if args.ppl_target > 0:
            assert not args.compress_kv_cache, "ppl_target is not supported when compressing kv_cache now"
            for layername, param_ratio in layers_min_ratio.items():
                raw_linear = module_dict[layername]
                info = linear_info[raw_linear]
                svd_linear = SVDLinear.from_linear(
                    raw_linear,
                    param_ratio=param_ratio,
                    alpha=args.alpha,
                    act_aware=args.act_aware,
                    sigma_fuse=args.sigma_fuse,
                    rank_align=args.rank_align,
                )
                setattr(info["father"], info["name"], svd_linear)
                tot_params += raw_linear.weight.numel()
                compress_params += raw_linear.weight.numel() * param_ratio
            ppl = evaluate_perplexity(model, input_ids, args.n_calib_samples)
            param_ratio = compress_params / tot_params
            msg = f"low={low} mid={mid}, high={high}, ppl={ppl}, param_ratio={param_ratio}"
            print(msg)
            if ppl < args.ppl_target:
                high = mid
            else:
                low = mid + 1
        else:
            for layername, param_ratio in layers_min_ratio.items():
                raw_linear = module_dict[layername]
                tot_params += raw_linear.weight.numel()
                compress_params += raw_linear.weight.numel() * param_ratio
            now_ratio = compress_params / tot_params
            if args.compress_kv_cache:
                # because param ratio is the params for ALinear+BLienar, so the rank ratio is param ratio/2
                now_ratio /= 2
            msg = f"low={low} mid={mid}, high={high}, now_ratio={now_ratio}, params=({compress_params}/{tot_params})"
            print(msg)
            if now_ratio > ratio_target:
                high = mid
            else:
                low = mid + 1

    print(f"=== Searching done, decomposing layers... ===")
    layers_min_ratio = {layername: default_param_ratio for layername in sensitivity_dict.keys()}
    for layername, param_ratio, ppl in sorted_sensitive_list[mid:]:
        if layers_min_ratio[layername] is None:
            layers_min_ratio[layername] = param_ratio
        else:
            layers_min_ratio[layername] = min(layers_min_ratio[layername], param_ratio)
    st = time.time()
    for layername, param_ratio in tqdm(layers_min_ratio.items()):
        # set ratio
        raw_linear = module_dict[layername]
        info = linear_info[raw_linear]
        if param_ratio == default_param_ratio:
            svd_linear = raw_linear
        else:
            svd_linear = SVDLinear.from_linear(
                raw_linear,
                param_ratio=param_ratio,
                alpha=args.alpha,
                act_aware=args.act_aware,
                sigma_fuse=args.sigma_fuse,
                rank_align=args.rank_align,
            )
            raw_linear.to("cpu")
        setattr(info["father"], info["name"], svd_linear)
        if svd_linear is not raw_linear:
            # `.to("cpu")` above frees VRAM but PARKS the original in host RAM,
            # where linear_info (keyed by this very module) and module_dict keep
            # it alive until the whole compression finishes. It accumulates over
            # every compressed matrix: ~98 GB on opt-66b, ~103 GB on
            # Llama-3.1-70B, against a 201 GB cgroup.
            #
            # Nothing reads the weight again: SVDLinear.from_linear consumes it
            # (svd_linear.py:47 `w = linear.weight.data.float()`) and its last
            # touch is the dtype lookup at :141, both before it returns. Release
            # the storage but keep the module object, because linear_info is
            # keyed on it. The BIAS is deliberately untouched -- svd_linear.py:76
            # holds it BY REFERENCE (`bias = linear.bias.data`), so freeing it
            # would corrupt the compressed layer.
            raw_linear.weight.data = torch.empty(
                0, dtype=raw_linear.weight.dtype, device="cpu")
        # print(f"decompose {info['full_name']} with ratio {param_ratio}")
    ed = time.time()
    print(f"decompose time: {ed-st}")


def binary_search_truncation_rank_optimize_scale(model, sensitivity_dict, calib_loader, args):
    module_dict = {name: module for name, module in model.named_modules()}
    full_name_dict = {module: name for name, module in model.named_modules()}
    linear_info = {}
    modules = [model]
    while len(modules) > 0:
        submodule = modules.pop()
        for name, raw_linear in submodule.named_children():
            if isinstance(raw_linear, nn.Linear):
                full_name = full_name_dict[raw_linear]
                linear_info[raw_linear] = {
                    "father": submodule,
                    "name": name,
                    "full_name": full_name,
                }
            else:
                modules.append(raw_linear)

    sensitivity_list = []
    for layername, v in sensitivity_dict.items():
        for ratio, ppl in v.items():
            sensitivity_list.append((layername, ratio, ppl))
    sorted_sensitive_list = sorted(sensitivity_list, key=lambda x: -x[2])

    # binary search
    high = len(sorted_sensitive_list) - 1
    low = 0
    assert args.ppl_target > 0 or args.param_ratio_target > 0

    input_ids = torch.cat([_["input_ids"] for _ in calib_loader], 0)
    while low < high:
        mid = (low + high) // 2
        layers_min_ratio = {layername: 1 for layername in sensitivity_dict.keys()}
        for layername, ratio, ppl in sorted_sensitive_list[mid:]:
            layers_min_ratio[layername] = min(layers_min_ratio[layername], ratio)
        tot_params = 0
        compress_params = 0
        if args.ppl_target > 0:
            for layername, ratio in layers_min_ratio.items():
                raw_linear = module_dict[layername]
                info = linear_info[raw_linear]
                svd_linear = GradSVDLinear.from_linear(
                    raw_linear,
                    param_ratio=ratio,
                    alpha=args.alpha,
                    act_aware=args.act_aware,
                    sigma_fuse=args.sigma_fuse,
                )
                setattr(info["father"], info["name"], svd_linear)
                tot_params += raw_linear.weight.numel()
                compress_params += raw_linear.weight.numel() * ratio
            ppl = evaluate_perplexity(model, input_ids, args.n_calib_samples)
            param_ratio = compress_params / tot_params
            msg = f"low={low} mid={mid}, high={high}, ppl={ppl}, param_ratio={param_ratio}"
            print(msg)
            if ppl < args.ppl_target:
                high = mid
            else:
                low = mid + 1
        else:
            for layername, ratio in layers_min_ratio.items():
                raw_linear = module_dict[layername]
                tot_params += raw_linear.weight.numel()
                compress_params += raw_linear.weight.numel() * ratio
            param_ratio = compress_params / tot_params
            msg = f"low={low} mid={mid}, high={high}, param_ratio={param_ratio}({compress_params}/{tot_params})"
            print(msg)
            if param_ratio > args.param_ratio_target:
                high = mid
            else:
                low = mid + 1

    print(f"Searching finished, decomposing layers...")
    layers_min_ratio = {layername: 1 for layername in sensitivity_dict.keys()}
    for layername, ratio, ppl in sorted_sensitive_list[mid:]:
        layers_min_ratio[layername] = min(layers_min_ratio[layername], ratio)
    for layername, ratio in tqdm(layers_min_ratio.items()):
        # set ratio
        raw_linear = module_dict[layername]
        info = linear_info[raw_linear]
        svd_linear = GradSVDLinear.from_linear(
            raw_linear,
            param_ratio=ratio,
            alpha=args.alpha,
            act_aware=args.act_aware,
            sigma_fuse=args.sigma_fuse,
        )
        setattr(info["father"], info["name"], svd_linear)
