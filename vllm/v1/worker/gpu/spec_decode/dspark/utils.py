# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch.nn as nn

from vllm.config import VllmConfig, replace
from vllm.model_executor.model_loader import get_model
from vllm.v1.worker.gpu.spec_decode.eagle.utils import (
    _should_share,
    get_target_lm_head,
    maybe_share_target_embed,
)


def load_dspark_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Module:
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config

    from vllm.compilation.backends import set_model_tag
    from vllm.model_executor.models.qwen3_dflash import dflash_has_any_non_causal
    from vllm.model_executor.models.utils import get_draft_quant_config

    # The draft model is loaded on the LAST PP rank only (the runner gates drafter
    # construction on get_pp_group().is_last_rank). instanttensor issues a world-group
    # all_reduce in _determine_io_params during load; with only one PP rank
    # participating, that collective has no partners on the other ranks and
    # deadlocks (NCCL watchdog timeout -> group teardown). get_model() reads
    # vllm_config.load_config (the TARGET's load-format), ignoring draft_load_config,
    # so honor an explicit draft_load_config here and force the default loader for
    # the draft whenever the effective load-format is instanttensor under PP>1.
    # (Port of 3058ae60f; the dflash/utils.py copy was already patched — this is
    # the dspark loader, which K3DSpark actually uses. Confirmed via py-spy on a
    # hung last-stage worker: stuck in instanttensor _determine_io_params.)
    from vllm.distributed.parallel_state import get_pp_group

    draft_load_config = speculative_config.draft_load_config or vllm_config.load_config
    if get_pp_group().world_size > 1 and draft_load_config.load_format == "instanttensor":
        draft_load_config = replace(draft_load_config, load_format="auto")
    draft_vllm_config = replace(
        vllm_config,
        load_config=draft_load_config,
        attention_config=replace(
            vllm_config.attention_config,
            use_non_causal=dflash_has_any_non_causal(draft_model_config.hf_config),
            backend=speculative_config.attention_backend,
        ),
        cache_config=(
            replace(
                vllm_config.cache_config,
                cache_dtype=speculative_config.kv_cache_dtype,
            )
            if speculative_config.kv_cache_dtype is not None
            else vllm_config.cache_config
        ),
    )
    # VllmConfig post-init restores the target's quant config because the target
    # config is retained for DSpark's target-layer metadata, so we must override it.
    draft_vllm_config.quant_config = get_draft_quant_config(vllm_config)

    with set_model_tag("dspark_head"):
        draft_model = get_model(
            vllm_config=draft_vllm_config, model_config=draft_model_config
        )

    target_language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    target_inner = target_language_model.model
    draft_inner = draft_model.model

    maybe_share_target_embed(draft_model, draft_inner, target_inner)

    target_lm_head = get_target_lm_head(target_model, target_language_model)
    draft_lm_head = getattr(draft_model, "lm_head", None)
    if target_lm_head is not None and _should_share(
        draft_model, "has_own_lm_head", draft_lm_head, target_lm_head
    ):
        if draft_lm_head is not None:
            del draft_model.lm_head
        draft_model.lm_head = target_lm_head

    return draft_model
