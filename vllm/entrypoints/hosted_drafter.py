# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hosted DFlash drafter: run vLLM itself as the standalone drafter service.

This is the "drafter-only" alternative to the hand-rolled PyTorch daemon in
``dflash-disagg/service/model.py``. Instead of re-implementing the DFlash
forward (fc / context-KV / non-causal block attention / partial-RoPE / sinks)
by hand -- which diverged and capped acceptance at AL~1.3 -- this boots a
*stripped* vLLM process that loads ONLY the draft model and reuses vLLM's REAL
spec-decode machinery:

  * ``DFlashQwen3ForCausalLM`` (model_executor/models/qwen3_dflash.py) for
    ``combine_hidden_states`` + ``precompute_and_store_context_kv`` + the
    decoder layers + the vLLM ``Attention`` op (sinks / sliding-window).
  * ``DFlashSpeculator`` (v1/worker/gpu/spec_decode/dflash/speculator.py) for
    ``prepare_dflash_inputs`` (the Triton kernel that lays out the bonus + mask
    query block and the context positions/slots) and ``propose()``.
  * vLLM's attention backend + paged KV cache + block tables, so the math is
    byte-identical to the co-located DFlash path (AL~2.58 / "count to 40" ~5.0).

It speaks the SAME NIXL wire protocol the existing daemon uses (see
``dflash-disagg/service/transport.py`` and the engine-side producer
``vllm/v1/worker/gpu/spec_decode/dflash/remote.py``), so a target running the
unmodified ``RemoteDFlashSpeculator`` can point ``draft_service_url`` at this
process with no engine change.

Boot it via the env flag (the entry point checks it):

    VLLM_HOSTED_DRAFTER=1 python -m vllm.entrypoints.hosted_drafter \
        --speculative-config '{"method":"dflash","model":"/models/.../dflash",
                               "num_speculative_tokens":7}' \
        --target-model XiaomiMiMo/MiMo-V2.5-Pro-FP4 \
        --port 5559 --max-model-len 131072 --max-num-seqs 32

Status: PROTOTYPE. The boot/KV-init harness (HostedDrafterRunner) is written to
mirror GPUModelRunner.initialize_kv_cache exactly; the wire loop reuses the
daemon's DrafterTransport verbatim. See the module-level NOTES for the parts
that still need validation against a live target.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from typing import Any

import numpy as np
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


# ----------------------------------------------------------------------------
# 1. Build the draft-only VllmConfig.
# ----------------------------------------------------------------------------
def build_drafter_vllm_config(
    speculative_config_json: str,
    target_model: str,
    max_model_len: int,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    dtype: str,
):
    """Construct a VllmConfig whose *primary* model is the DFlash draft head.

    The trick: ``DFlashSpeculator`` reads its draft model from
    ``vllm_config.speculative_config.draft_model_config``, but the speculator's
    attention layers, KV-cache spec, and ``get_layers_from_vllm_config`` all
    walk the layers registered on the live ``model``. In a normal engine the
    *target* is the live model and the draft layers are appended after it
    (DFlashQwen3Model uses ``start_layer_id=target_layer_num`` so its layer
    prefixes don't collide). Here there is no target, so we register the draft
    head AS the engine's model_config too -- but we still need the target's
    hidden_size / num_layers / vocab for aux_width + lm_head sharing.

    We therefore keep the target's ModelConfig for ``model_config`` (so
    ``get_num_layers`` / vocab / hidden are correct and the draft head's
    ``start_layer_id`` lands past the target's layer count, exactly as in the
    engine), and the draft repo for ``speculative_config.draft_model_config``.
    Only the draft head is ever actually instantiated (load_target_model is
    skipped -- see HostedDrafterRunner.load_model).
    """
    from vllm.engine.arg_utils import EngineArgs

    # EngineArgs gives us a fully-resolved VllmConfig (model_config, cache,
    # parallel, scheduler, compilation, speculative) with all the defaults the
    # engine would compute -- crucially the SpeculativeConfig parsing that turns
    # the JSON into a draft_model_config + num_speculative_tokens. We force
    # single-process (TP=1) here: a hosted drafter is one GPU; the *target* may
    # be TP=8 but it only ships rank-0's (already all-reduced) aux to us.
    sc = json.loads(speculative_config_json)
    # The engine's RemoteDFlashSpeculator never sets draft_service_url on US; we
    # ARE the service. Strip it so init picks the local DFlashSpeculator path.
    sc.pop("draft_service_url", None)
    sc.pop("draft_service_timeout_ms", None)

    engine_args = EngineArgs(
        model=target_model,
        speculative_config=sc,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        dtype=dtype,
        tensor_parallel_size=1,
        enforce_eager=True,  # CG capture for the drafter-only harness is future work
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
        enable_prefix_caching=False,
        load_format="auto",
    )
    vllm_config = engine_args.create_engine_config()
    assert vllm_config.speculative_config is not None
    assert vllm_config.speculative_config.method == "dflash", (
        "hosted drafter only supports method=dflash for now"
    )
    return vllm_config


# ----------------------------------------------------------------------------
# 2. The stripped runner: draft model + KV cache + block tables, no target.
# ----------------------------------------------------------------------------
class HostedDrafterRunner:
    """A minimal GPUModelRunner stand-in that owns ONLY the draft model.

    Mirrors the subset of GPUModelRunner the DFlashSpeculator depends on:
    load_model -> set_attn -> initialize_kv_cache, plus a per-step
    block-allocation + InputBatch construction that the engine's scheduler
    normally provides.
    """

    def __init__(self, vllm_config, device: torch.device):
        self.vllm_config = vllm_config
        self.device = device
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.compilation_config = vllm_config.compilation_config

        self.max_num_reqs = self.scheduler_config.max_num_seqs
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        self.max_model_len = self.model_config.max_model_len

        # Per-request drafter-side block allocator (no scheduler here).
        self._free_blocks: list[int] = []
        self._req_blocks: dict[str, list[int]] = {}
        self._req_index: dict[str, int] = {}  # req_id -> stable batch row
        self._free_indices: list[int] = list(range(self.max_num_reqs))

    # -- model load ---------------------------------------------------------
    def load_model(self) -> None:
        """Instantiate ONLY the draft head and bolt on target embed/lm_head.

        We do NOT call model_loader.load_model on the target (that would pull
        the full ~570 GB MiMo). We build the draft head with get_model (the same
        call load_dflash_model makes) and then provide embed_tokens + lm_head
        from the target checkpoint shard, replacing the sharing that
        load_dflash_model normally does from the in-process target model.
        """
        from vllm.config import replace
        from vllm.model_executor.model_loader import get_model
        from vllm.v1.worker.gpu.spec_decode.dflash.utils import get_dflash_causal

        sc = self.vllm_config.speculative_config
        draft_model_config = sc.draft_model_config
        causal = get_dflash_causal(draft_model_config)
        draft_vllm_config = replace(
            self.vllm_config,
            attention_config=replace(
                self.vllm_config.attention_config, use_non_causal=not causal
            ),
        )
        from vllm.compilation.backends import set_model_tag

        with set_model_tag("dflash_head"):
            self.model = get_model(
                vllm_config=draft_vllm_config, model_config=draft_model_config
            )

        self._attach_target_embed_lm_head()

        # Construct the speculator and run the load_model bookkeeping that
        # DraftModelSpeculator.load_model does (compute draft_attn_layer_names),
        # but WITHOUT a target model -- all attention layers in the config are
        # the draft's, so draft_attn_layer_names == all_attn_layers.
        from vllm.config import get_layers_from_vllm_config
        from vllm.model_executor.layers.attention_layer_base import (
            AttentionLayerBase,
        )
        from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
            DFlashSpeculator,
        )

        self.speculator = DFlashSpeculator(self.vllm_config, self.device)
        self.speculator.model = self.model
        all_attn_layers = set(
            get_layers_from_vllm_config(
                self.vllm_config, AttentionLayerBase
            ).keys()
        )
        # No target was loaded, so every registered attention layer is a draft
        # layer. (The draft head's Attention modules register themselves on the
        # global static_forward_context when get_model instantiates them.)
        self.speculator.draft_attn_layer_names = all_attn_layers
        logger.info(
            "hosted drafter: %d draft attention layers: %s",
            len(all_attn_layers), sorted(all_attn_layers),
        )

    def _attach_target_embed_lm_head(self) -> None:
        """Pull embed_tokens + lm_head off the target checkpoint.

        The MiMo DFlash checkpoint carries NO embed_tokens / lm_head / d2t (the
        co-located engine shares them from the in-process target -- see
        load_dflash_model). We replicate the daemon's targeted shard download
        (service/model.py:_load_target_embed_lm_head) and bind the tensors onto
        the draft model's embed_tokens / lm_head modules.
        """
        # Reuse the daemon's proven loader to avoid drift.
        import importlib.util
        import sys

        daemon_model_path = os.environ.get(
            "DFLASH_DAEMON_MODEL_PY",
            "/home/ian/sandbox/localvllm/dflash-disagg/service/model.py",
        )
        target_repo = os.environ["HOSTED_DRAFTER_TARGET_MODEL"]
        hidden = self.model.config.hidden_size

        if os.path.exists(daemon_model_path):
            spec = importlib.util.spec_from_file_location(
                "_dflash_daemon_model", daemon_model_path
            )
            mod = importlib.util.module_from_spec(spec)
            sys.modules["_dflash_daemon_model"] = mod
            spec.loader.exec_module(mod)
            embed, head = mod._load_target_embed_lm_head(
                target_repo, hidden, self.model_config.dtype, self.device
            )
        else:
            embed, head = _fallback_load_target_embed_lm_head(
                target_repo, hidden, self.model_config.dtype, self.device
            )

        inner = self.model.model
        # The draft head built embed_tokens / lm_head as empty parallel modules
        # (VocabParallelEmbedding / ParallelLMHead). Copy the target weights in.
        # The draft config's vocab (e.g. 152064) may differ from the target's
        # padded embed/lm_head (e.g. 152576). The extra rows are vocab padding
        # (never real tokens), so copy the overlapping [:min] rows.
        def _copy_vocab(dst, src):
            n = min(dst.shape[0], src.shape[0])
            dst[:n].copy_(src[:n].to(dst.dtype))
        with torch.no_grad():
            if getattr(inner, "embed_tokens", None) is not None:
                _copy_vocab(inner.embed_tokens.weight.data, embed)
            if getattr(self.model, "lm_head", None) is not None:
                _copy_vocab(self.model.lm_head.weight.data, head)
        logger.info(
            "hosted drafter: bound target embed_tokens%s + lm_head%s from %s",
            tuple(embed.shape), tuple(head.shape), target_repo,
        )

    # -- KV cache / attention backend / block tables ------------------------
    def initialize_kv_cache(self, num_gpu_blocks: int | None = None) -> None:
        """Stand up the draft KV cache + attention backend + block tables.

        A line-for-line analogue of GPUModelRunner.initialize_kv_cache, minus
        the target model and minus the encoder-decoder / mamba branches. The
        only judgement call is num_gpu_blocks: with no target to profile against
        we either accept an explicit count or size for the full max_model_len of
        a single request (the drafter's per-request context never exceeds the
        target's, and only TP-rank-0 aux is shipped, so a single-stream-worth of
        blocks per request is sufficient).
        """
        from vllm.utils.math_utils import cdiv
        from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
        from vllm.v1.worker.gpu.attn_utils import (
            get_kv_cache_spec,
            init_attn_backend,
            init_kv_cache,
        )
        from vllm.v1.worker.gpu.block_table import BlockTables

        kv_cache_spec = get_kv_cache_spec(self.vllm_config)
        assert kv_cache_spec, (
            "no KV cache spec found -- the draft head's Attention layers did "
            "not register. Did get_model actually instantiate the draft model?"
        )

        # Size the cache. The drafter is tiny (~5.5 GiB weights); the rest of
        # the GPU is KV. Either take an explicit block count or compute one that
        # covers max_model_len * max_num_reqs.
        if num_gpu_blocks is None:
            available = self._estimate_kv_bytes()
        else:
            # Reverse out the byte budget from a requested block count.
            per_block = sum(
                spec.page_size_bytes for spec in kv_cache_spec.values()
            )
            available = num_gpu_blocks * per_block

        kv_cache_configs = get_kv_cache_configs(
            self.vllm_config, [kv_cache_spec], [available]
        )
        kv_cache_config = kv_cache_configs[0]
        self.kv_cache_config = kv_cache_config
        logger.info(
            "hosted drafter KV cache: %d blocks across %d groups",
            kv_cache_config.num_blocks, len(kv_cache_config.kv_cache_groups),
        )

        block_sizes = []
        max_num_blocks_per_group = []
        for kv_cache_group in kv_cache_config.kv_cache_groups:
            spec = kv_cache_group.kv_cache_spec
            block_sizes.append(spec.block_size)
            max_num_blocks = cdiv(self.max_model_len, spec.block_size)
            if spec.block_size <= 128:
                alignment = 128 // spec.block_size
                max_num_blocks = cdiv(max_num_blocks, alignment) * alignment
            max_num_blocks_per_group.append(max_num_blocks)

        self.attn_groups, _, self.kernel_block_sizes = init_attn_backend(
            self.kv_cache_config, self.vllm_config, self.device
        )
        self.block_tables = BlockTables(
            block_sizes=block_sizes,
            max_num_reqs=self.max_num_reqs,
            max_num_batched_tokens=self.max_num_tokens,
            max_num_blocks_per_group=max_num_blocks_per_group,
            device=self.device,
            kernel_block_sizes=self.kernel_block_sizes,
        )

        # ModelState is only *stored* by set_attn (never read on the DFlash
        # propose path), and init_model_state would inspect the TARGET's
        # architecture (model_config is the target) and build target-specific
        # state we don't have. A placeholder is sufficient and avoids that.
        self.model_state = None
        from vllm.config.compilation import CUDAGraphMode

        # enforce_eager=True in build_drafter_vllm_config => no CG capture.
        cudagraph_mode = self.compilation_config.cudagraph_mode or CUDAGraphMode.NONE
        self.speculator.init_cudagraph_manager(cudagraph_mode)
        self.speculator.set_attn(
            self.model_state, self.kv_cache_config, self.block_tables
        )

        self.kv_caches: list[torch.Tensor] = []
        init_kv_cache(
            self.kv_caches,
            self.compilation_config.static_forward_context,
            self.kv_cache_config,
            self.attn_groups,
            self.device,
            self.cache_config.cache_dtype,
            self.kernel_block_sizes,
            self.vllm_config,
        )

        # Build the free-block pool from whatever the cache sized to. Block 0 is
        # reserved as the null/padding block (PAD_SLOT_ID writes never land
        # there because prepare_dflash_inputs masks them).
        self.block_size = block_sizes[0]
        self._free_blocks = list(range(1, kv_cache_config.num_blocks))
        logger.info(
            "hosted drafter ready: block_size=%d free_blocks=%d",
            self.block_size, len(self._free_blocks),
        )

    def _estimate_kv_bytes(self) -> int:
        free, total = torch.cuda.mem_get_info()
        # Leave headroom for activations + the staging/ring buffers + fragmentation.
        util = float(os.environ.get("HOSTED_DRAFTER_GPU_UTIL", "0.85"))
        return int(free * util)

    # -- per-request block management --------------------------------------
    def _row_for(self, req_id: str) -> int:
        idx = self._req_index.get(req_id)
        if idx is None:
            if not self._free_indices:
                raise RuntimeError("hosted drafter out of request slots")
            idx = self._free_indices.pop(0)
            self._req_index[req_id] = idx
            self._req_blocks[req_id] = []
        return idx

    def _ensure_blocks(self, req_id: str, num_positions: int) -> None:
        """Make sure req has enough blocks to cover positions [0, num_positions).

        The block table is indexed by *absolute position* in the slot-mapping
        kernel (slot = block_table[pos // block_size] * block_size + pos %
        block_size), so we must back every block index up to the highest
        position, even ones reused from a cached prefix the target never ships
        (those slots simply stay zero, which precompute_and_store_context_kv +
        the base= window logic already handle on the engine side).
        """
        idx = self._row_for(req_id)
        needed = (num_positions + self.block_size - 1) // self.block_size
        have = self.block_tables.num_blocks.np[
            self.speculator.draft_kv_cache_group_id, idx
        ]
        if have >= needed:
            return
        add = needed - have
        if add > len(self._free_blocks):
            raise RuntimeError(
                f"hosted drafter OOM blocks: need {add}, have "
                f"{len(self._free_blocks)} (raise max_model_len budget or "
                f"lower it)"
            )
        new_ids = [self._free_blocks.pop() for _ in range(add)]
        self.block_tables.append_block_ids(idx, (new_ids,), overwrite=False)
        self.block_tables.apply_staged_writes()

    def free(self, req_ids: list[str]) -> None:
        for rid in req_ids:
            idx = self._req_index.pop(rid, None)
            blocks = self._req_blocks.pop(rid, None)
            if blocks:
                self._free_blocks.extend(blocks)
            if idx is not None:
                self._free_indices.append(idx)
                # Zero the freed row's block-count so a future reuse starts clean.
                self.block_tables.num_blocks.np[:, idx] = 0

    def reset(self) -> None:
        for rid in list(self._req_index.keys()):
            self.free([rid])


# ----------------------------------------------------------------------------
# 3. Wire-step -> InputBatch -> DFlashSpeculator.propose() -> draft ids.
# ----------------------------------------------------------------------------
class HostedDrafterStepRunner:
    """Translates one wire step header + aux ring rows into a call to the REAL
    DFlashSpeculator.propose(), and returns draft token ids [num_reqs, k]."""

    def __init__(self, runner: HostedDrafterRunner):
        self.r = runner
        self.device = runner.device
        self.spec = runner.speculator
        self.k = self.spec.num_speculative_steps
        self.dtype = runner.model_config.dtype

        # Reusable scratch the engine-side speculator's propose() expects.
        self._max_reqs = runner.max_num_reqs
        self._last_sampled = torch.zeros(
            self._max_reqs, dtype=torch.int64, device=self.device
        )
        self._next_prefill = torch.zeros(
            self._max_reqs, dtype=torch.int64, device=self.device
        )
        self._temperature = torch.zeros(
            self._max_reqs, dtype=torch.float32, device=self.device
        )
        self._seeds = torch.zeros(
            self._max_reqs, dtype=torch.int64, device=self.device
        )

    @torch.inference_mode()
    def run(
        self, header: dict[str, Any], aux: torch.Tensor
    ) -> torch.Tensor:
        """header: the msgpack step header from the target (reqs[], finished[]).
        aux: [num_tokens, aux_width] target aux rows pulled from the ring.

        Returns ids [num_reqs, k] int64 (target-vocab) to WRITE back.
        """
        reqs = header["reqs"]
        num_reqs = len(reqs)
        self.r.free(header.get("finished", []))

        # Lay out the flattened scheduled-token arrays from per-request ctx_pos.
        positions_list: list[int] = []
        qsl = [0]
        num_sched = []
        num_computed = []
        num_rejected = []
        idx_mapping_np = np.empty(num_reqs, dtype=np.int32)
        req_ids: list[str] = []
        for i, rq in enumerate(reqs):
            ctx_pos = rq["ctx_pos"]  # absolute positions of the scheduled tokens
            n = rq["n_sched"]
            assert len(ctx_pos) == n, (
                f"ctx_pos len {len(ctx_pos)} != n_sched {n}"
            )
            positions_list.extend(ctx_pos)
            qsl.append(qsl[-1] + n)
            num_sched.append(n)
            num_rejected.append(rq["n_rej"])
            # num_computed = first position of this request (start_pos).
            num_computed.append(rq["start_pos"])
            row = self.r._row_for(rq["id"])
            idx_mapping_np[i] = row
            req_ids.append(rq["id"])
            # Bonus token goes in last_sampled[row]; mark n_samp>0 so the kernel
            # reads last_sampled (not next_prefill).
            self._last_sampled[row] = rq["bonus"]
            # Allocate blocks to cover this request's highest position + the
            # query block (k+1 query tokens past the last context position).
            highest = (max(ctx_pos) if ctx_pos else rq["start_pos"]) + self.k + 2
            self.r._ensure_blocks(rq["id"], highest)

        num_tokens = qsl[-1]
        positions = torch.tensor(
            positions_list, dtype=torch.int64, device=self.device
        )
        query_start_loc_np = np.array(qsl, dtype=np.int32)
        query_start_loc = torch.tensor(
            qsl, dtype=torch.int32, device=self.device
        )
        idx_mapping = torch.from_numpy(idx_mapping_np).to(self.device)

        num_sampled_t = torch.ones(
            num_reqs, dtype=torch.int64, device=self.device
        )
        num_rejected_t = torch.tensor(
            num_rejected, dtype=torch.int64, device=self.device
        )

        # combine_hidden_states wants the cat'd aux already (the target ships the
        # already-concatenated aux rows). DFlashSpeculator.propose expects a
        # *list* of per-layer aux tensors that it re-concatenates, so split the
        # ring rows back into the per-layer chunks.
        per_layer = self._split_aux(aux[:num_tokens])

        # Gather each request's staged block table into input_block_tables,
        # indexed by batch row (req_idx). prepare_dflash_inputs + the draft
        # attention metadata read input_block_tables[group][req_idx], NOT the
        # staged per-req-state tables, so this MUST run every step (the engine
        # runner does the same before speculator.propose -- model_runner.py
        # gather_block_tables()).
        self.r.block_tables.gather_block_tables(idx_mapping, num_reqs)

        input_batch = self._make_input_batch(
            req_ids=req_ids,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            idx_mapping=idx_mapping,
            idx_mapping_np=idx_mapping_np,
            positions=positions,
            query_start_loc=query_start_loc,
            query_start_loc_np=query_start_loc_np,
            num_scheduled_tokens=np.array(num_sched, dtype=np.int32),
            num_computed_tokens_np=np.array(num_computed, dtype=np.int32),
        )

        draft_tokens = self.spec.propose(
            input_batch=input_batch,
            attn_metadata={},          # built internally by the speculator
            slot_mappings={},          # built internally by the speculator
            last_hidden_states=None,   # unused when aux_hidden_states is given
            aux_hidden_states=per_layer,
            num_sampled=num_sampled_t,
            num_rejected=num_rejected_t,
            last_sampled=self._last_sampled,
            next_prefill_tokens=self._next_prefill,
            temperature=self._temperature,
            seeds=self._seeds,
        )
        return draft_tokens  # [num_reqs, k]

    def _split_aux(self, aux: torch.Tensor) -> list[torch.Tensor]:
        sc = self.r.vllm_config.speculative_config
        dflash_cfg = (
            getattr(sc.draft_model_config.hf_config, "dflash_config", None) or {}
        )
        n_layers = len(dflash_cfg.get("target_layer_ids") or [])
        assert n_layers > 0
        per = aux.shape[-1] // n_layers
        return [aux[:, j * per:(j + 1) * per].contiguous() for j in range(n_layers)]

    def _make_input_batch(self, **kw):
        from vllm.v1.worker.gpu.input_batch import InputBatch

        num_reqs = kw["num_reqs"]
        num_tokens = kw["num_tokens"]
        device = self.device
        qsl = kw["query_start_loc"]
        # The speculator only reads a subset of InputBatch; fill the rest with
        # safe placeholders that satisfy shape/dtype contracts.
        seq_lens = torch.tensor(
            [kw["num_computed_tokens_np"][i] + kw["num_scheduled_tokens"][i]
             for i in range(num_reqs)],
            dtype=torch.int32, device=device,
        )
        return InputBatch(
            req_ids=kw["req_ids"],
            num_reqs=num_reqs,
            num_reqs_after_padding=num_reqs,
            idx_mapping=kw["idx_mapping"],
            idx_mapping_np=kw["idx_mapping_np"],
            expanded_idx_mapping=kw["idx_mapping"],
            expanded_local_pos=torch.zeros(
                num_reqs, dtype=torch.int32, device=device
            ),
            num_scheduled_tokens=kw["num_scheduled_tokens"],
            num_tokens=num_tokens,
            num_tokens_after_padding=num_tokens,
            num_draft_tokens=0,
            num_draft_tokens_per_req=None,
            query_start_loc=qsl,
            query_start_loc_np=kw["query_start_loc_np"],
            seq_lens=seq_lens,
            seq_lens_cpu_upper_bound=torch.from_numpy(
                (kw["num_computed_tokens_np"] + kw["num_scheduled_tokens"]).astype(
                    np.int32
                )
            ),
            dcp_local_seq_lens=None,
            num_computed_tokens_np=kw["num_computed_tokens_np"],
            prefill_len_np=np.zeros(num_reqs, dtype=np.int32),
            num_computed_prefill_tokens_np=np.zeros(num_reqs, dtype=np.int32),
            is_prefilling_np=np.zeros(num_reqs, dtype=np.bool_),
            max_seq_len_np=None,
            input_ids=torch.zeros(num_tokens, dtype=torch.int32, device=device),
            positions=kw["positions"],
            logits_indices=qsl[1:] - 1,
            cu_num_logits=torch.arange(
                num_reqs + 1, dtype=torch.int32, device=device
            ),
            cu_num_logits_np=np.arange(num_reqs + 1, dtype=np.int32),
            has_structured_output_reqs=False,
        )


# ----------------------------------------------------------------------------
# 4. Fallback target embed/lm_head loader (used if the daemon module is absent).
# ----------------------------------------------------------------------------
def _fallback_load_target_embed_lm_head(target_repo, hidden, dtype, device):
    import json as _json

    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    embed_names = [
        "model.embed_tokens.weight",
        "model.language_model.embed_tokens.weight",
        "language_model.model.embed_tokens.weight",
    ]
    head_names = [
        "lm_head.weight",
        "model.lm_head.weight",
        "language_model.lm_head.weight",
    ]
    snap = snapshot_download(
        target_repo, allow_patterns=["*.safetensors.index.json", "config.json"]
    )
    idx_path = os.path.join(snap, "model.safetensors.index.json")
    wmap = _json.load(open(idx_path))["weight_map"]
    shards = sorted({wmap[n] for n in (*embed_names, *head_names) if n in wmap})
    snap = snapshot_download(target_repo, allow_patterns=shards)

    def fetch(name):
        if name not in wmap:
            return None
        with safe_open(os.path.join(snap, wmap[name]), framework="pt") as f:
            return f.get_tensor(name)

    embed = next((t for n in embed_names if (t := fetch(n)) is not None), None)
    head = next((t for n in head_names if (t := fetch(n)) is not None), None)
    if head is None:
        head = embed
    return embed.to(device, dtype), head.to(device, dtype)


# ----------------------------------------------------------------------------
# 5. Serve loop (reuses the daemon's DrafterTransport verbatim).
# ----------------------------------------------------------------------------
def serve(args: argparse.Namespace) -> None:
    import sys

    # Make the daemon's transport importable (it lives outside the vLLM tree).
    disagg_root = os.environ.get(
        "DFLASH_DISAGG_ROOT", "/home/ian/sandbox/localvllm/dflash-disagg"
    )
    if disagg_root not in sys.path:
        sys.path.insert(0, disagg_root)
    from service.transport import DrafterTransport  # type: ignore

    os.environ["HOSTED_DRAFTER_TARGET_MODEL"] = args.target_model
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    vllm_config = build_drafter_vllm_config(
        args.speculative_config,
        args.target_model,
        args.max_model_len,
        args.max_num_seqs,
        args.max_num_batched_tokens,
        args.dtype,
    )

    # vLLM's parallel layers + CustomOp dispatch call get_current_vllm_config()
    # at construction AND forward time, so keep the config context open for the
    # whole drafter-only process lifetime (never exited -- this is a server).
    from vllm.config import set_current_vllm_config
    # Keep a live reference: a @contextmanager whose object is GC'd runs its
    # finally (resetting the config). _cfg_cm lives for serve()'s lifetime.
    _cfg_cm = set_current_vllm_config(vllm_config)
    _cfg_cm.__enter__()

    # Initialize the (single-rank) distributed env vLLM's layers expect even at
    # TP=1 (get_tp_group / tensor_model_parallel_rank are referenced on load).
    _init_single_rank_dist()

    runner = HostedDrafterRunner(vllm_config, device)
    logger.info("hosted drafter: loading draft model...")
    runner.load_model()
    logger.info("hosted drafter: initializing KV cache...")
    runner.initialize_kv_cache(num_gpu_blocks=args.num_gpu_blocks)
    step = HostedDrafterStepRunner(runner)

    # Build the spec handshake the target validates against (mirrors
    # daemon DrafterSpec fields the transport ships).
    sc = vllm_config.speculative_config
    draft_hf = sc.draft_model_config.hf_config
    dflash_cfg = getattr(draft_hf, "dflash_config", None) or {}
    target_layer_ids = list(dflash_cfg.get("target_layer_ids") or [])
    per_layer_hidden = getattr(
        draft_hf, "target_hidden_size", draft_hf.hidden_size
    )
    aux_width = len(target_layer_ids) * per_layer_hidden
    spec_payload = {
        "num_aux_layers": len(target_layer_ids),
        "target_layer_ids": target_layer_ids,
        "aux_width": aux_width,
        "hidden_size": draft_hf.hidden_size,
        "mask_token_id": dflash_cfg.get("mask_token_id"),
        "block_size": dflash_cfg.get("block_size"),
        "num_spec_tokens": sc.num_speculative_tokens,
        "vocab_size": sc.draft_model_config.get_vocab_size(),
        "dtype": str(vllm_config.model_config.dtype).split(".")[-1],
    }

    tx = DrafterTransport(
        spec_payload=spec_payload,
        aux_width=aux_width,
        ring_tokens=args.ring_tokens,
        max_reqs=args.max_num_seqs,
        num_spec=sc.num_speculative_tokens,
        device=device,
        dtype=vllm_config.model_config.dtype,
        port=args.port,
    )
    tx.start_bootstrap_listener(spec_payload)
    logger.info("hosted drafter serving on :%d (aux_width=%d k=%d)",
                args.port, aux_width, sc.num_speculative_tokens)

    steps = 0
    while True:
        try:
            hdr = tx.wait_step()
        except Exception:
            logger.exception("transport poll error; continuing")
            time.sleep(0.05)
            continue
        kind = hdr.get("kind", "step")
        if kind == "reconnect":
            runner.reset()
            logger.info("target connected/reconnected; request state reset")
            continue
        if kind == "reset":
            runner.reset()
            continue
        if kind == "free":
            runner.free(hdr.get("req_ids", []))
            continue

        reqs = hdr["reqs"]
        try:
            aux = tx.aux_rows(hdr["row_off"], hdr["num_tokens"])
            ids = step.run(hdr, aux)  # [num_reqs, k]
            n = min(ids.shape[0], tx.ids_out.shape[0])
            tx.ids_out[:n] = ids[:n]
            if n < len(reqs):
                tx.ids_out[n:len(reqs)].zero_()
            torch.cuda.synchronize()
        except Exception:
            logger.exception(
                "step %s failed; replying zero drafts", hdr.get("step_id")
            )
            tx.ids_out[: len(reqs)].zero_()
            torch.cuda.synchronize()
        try:
            tx.send_ids(len(reqs), hdr["step_id"])
        except Exception:
            logger.exception("send_ids failed for step %s", hdr.get("step_id"))
        steps += 1
        if args.log_steps or steps % 200 == 0:
            logger.info("step %d: reqs=%d tokens=%d",
                        hdr["step_id"], len(reqs), hdr["num_tokens"])


def _init_single_rank_dist() -> None:
    """Bring up a 1-rank TP/PP world so vLLM's parallel layers load at TP=1."""
    import vllm.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", os.environ.get("HOSTED_DRAFTER_DIST_PORT", "29555"))
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="nccl", world_size=1, rank=0
        )
    dist.init_distributed_environment(
        world_size=1, rank=0, local_rank=0
    )
    dist.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Hosted DFlash drafter (vLLM-as-drafter-service)"
    )
    p.add_argument(
        "--speculative-config", required=True,
        help='JSON, same shape as the target engine\'s speculative-config '
             '(method/model/num_speculative_tokens). draft_service_url is '
             'ignored here -- this process IS the service.',
    )
    p.add_argument(
        "--target-model", required=True,
        help="Target repo/path -- used only for embed_tokens + lm_head + "
             "hidden/vocab/layer-count metadata (NOT loaded as a full model).",
    )
    p.add_argument("--port", type=int, default=5559)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--max-model-len", type=int, default=131072)
    p.add_argument("--max-num-seqs", type=int, default=32)
    p.add_argument("--max-num-batched-tokens", type=int, default=4096)
    p.add_argument("--ring-tokens", type=int, default=16384)
    p.add_argument(
        "--num-gpu-blocks", type=int, default=None,
        help="Override the KV block count (else sized from free GPU memory).",
    )
    p.add_argument("--log-steps", action="store_true")
    args = p.parse_args()
    serve(args)


if __name__ == "__main__":
    if os.environ.get("VLLM_HOSTED_DRAFTER", "1") == "0":
        raise SystemExit(
            "VLLM_HOSTED_DRAFTER=0 set; refusing to start the hosted drafter."
        )
    main()
