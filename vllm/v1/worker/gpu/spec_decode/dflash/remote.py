# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Remote DFlash speculator: the draft model runs in a separate process or on
a separate machine ("drafter service"); this class ships the target's aux
hidden states to it over NIXL each step and returns the draft token ids the
service RDMA-WRITEs back.

Wire contract (token-ids-only; greedy drafting):
  bootstrap (plain TCP, once): exchange NIXL agent metadata + buffer
    descriptors + a compat handshake (aux width, mask token, block size).
  per step: RDMA-WRITE staged aux rows into the service's ring buffer with the
    msgpack step header as the transfer notification; the service WRITEs
    int64 draft ids [num_reqs, num_spec] into our registered ``draft_tokens``
    buffer and notifies ``ack:<step_id>``.

Failure mode: any timeout/error degrades to zero drafts for the step (the
rejection sampler guarantees output correctness regardless of draft content);
a circuit breaker stops calling an unreachable service and retries periodically.

TP>1: aux hidden states are replicated across TP ranks after the final
all-reduce, so only TP rank 0 holds the NIXL connection, ships its local copy,
and broadcasts the returned draft ids [num_reqs, k] to the other ranks each
step (one tiny int64 broadcast). The other ranks allocate nothing but the
broadcast target.
"""

import pickle
import socket
import struct
import time
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import get_tensor_model_parallel_rank, get_tp_group
from vllm.logger import init_logger
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator

logger = init_logger(__name__)

_CONNECT_TIMEOUT_S = 5.0
_BREAKER_THRESHOLD = 3
_BREAKER_RETRY_S = 30.0


def _send_msg(sock: socket.socket, obj: Any) -> None:
    data = pickle.dumps(obj)
    sock.sendall(struct.pack("!I", len(data)) + data)


def _recv_msg(sock: socket.socket) -> Any:
    hdr = _recv_exact(sock, 4)
    return pickle.loads(_recv_exact(sock, struct.unpack("!I", hdr)[0]))


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("drafter bootstrap peer closed")
        buf += chunk
    return buf


class RemoteDFlashSpeculator(BaseSpeculator):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self.vllm_config = vllm_config
        self.device = device
        sc = vllm_config.speculative_config
        assert sc is not None and sc.draft_service_url is not None

        if vllm_config.parallel_config.data_parallel_size != 1:
            raise NotImplementedError(
                "Remote DFlash drafter does not support data_parallel_size>1 "
                "yet (each DP replica would need its own drafter service)."
            )
        # TP>1: rank 0 owns all service I/O; the other ranks only hold the
        # draft_tokens broadcast target (see module docstring).
        self.tp_group = get_tp_group()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = self.tp_group.world_size

        host, _, port = sc.draft_service_url.rpartition(":")
        self.host = host.removeprefix("tcp://") or "127.0.0.1"
        self.port = int(port)
        self.timeout_s = sc.draft_service_timeout_ms / 1000.0

        self.num_speculative_steps = sc.num_speculative_tokens
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.dtype = vllm_config.model_config.dtype

        self.method = sc.method
        draft_hf = sc.draft_model_config.hf_config
        if self.method == "eagle3":
            eagle_cfg = getattr(draft_hf, "eagle_config", None) or {}
            target_layer_ids = (
                eagle_cfg.get("eagle_aux_hidden_state_layer_ids") or []
            )
            if not target_layer_ids:
                raise ValueError(
                    "draft model config has no eagle_config."
                    "eagle_aux_hidden_state_layer_ids; not an EAGLE3 head?"
                )
        else:
            dflash_cfg = getattr(draft_hf, "dflash_config", None) or {}
            target_layer_ids = dflash_cfg.get("target_layer_ids") or []
            if not target_layer_ids:
                raise ValueError(
                    "draft model config has no dflash_config.target_layer_ids;"
                    " not a DFlash drafter?"
                )
        per_layer_hidden = getattr(
            draft_hf, "target_hidden_size", draft_hf.hidden_size
        )
        self.aux_width = len(target_layer_ids) * per_layer_hidden

        # Interface attributes the runner reads.
        self.supports_mm_inputs = False
        self.draft_logits: torch.Tensor | None = None

        # The drafter service WRITEs draft ids here (NIXL-registered on rank
        # 0); ranks >0 receive it via the per-step TP broadcast.
        self.draft_tokens = torch.zeros(
            self.max_num_reqs,
            self.num_speculative_steps,
            dtype=torch.int64,
            device=device,
        )
        if self.tp_rank == 0:
            # Aux states are staged here before the RDMA WRITE
            # (NIXL-registered).
            self.staging = torch.zeros(
                self.max_num_tokens,
                self.aux_width,
                dtype=self.dtype,
                device=device,
            )
            # Small-int D2H scratch: rows = num_sampled, num_rejected, bonus.
            self._scratch_gpu = torch.zeros(
                3, self.max_num_reqs, dtype=torch.int64, device=device
            )
            self._scratch_cpu = torch.zeros(
                3, self.max_num_reqs, dtype=torch.int64, pin_memory=True
            )
            self._copy_event = torch.cuda.Event()
        else:
            self.staging = None
            self._scratch_gpu = None
            self._scratch_cpu = None
            self._copy_event = None

        # NIXL state (lazy connect — the service may start after the engine).
        self._agent = None
        self._peer: str | None = None
        self._ring_base = 0
        self._ring_tokens = 0
        # NOTE: derived from the dtype, not from `staging` — staging is only
        # allocated on TP rank 0 and this runs on every rank.
        self._aux_row_bytes = self.aux_width * self.dtype.itemsize
        self._row_off = 0
        self._step_id = 0
        self._pending_finished: set[str] = set()

        # Circuit breaker.
        self._consecutive_failures = 0
        self._breaker_open_until = 0.0

        # Step metrics (logged periodically).
        self._steps_ok = 0
        self._steps_failed = 0
        self._rpc_ms_sum = 0.0

        if self.tp_rank == 0:
            logger.info(
                "Remote DFlash speculator: service=%s:%d aux_width=%d k=%d "
                "tp_size=%d (draft model is NOT loaded in this process; "
                "rank 0 transfers, then broadcasts draft ids)",
                self.host, self.port, self.aux_width,
                self.num_speculative_steps, self.tp_size,
            )

    # ------------------------------------------------------------------
    # Runner interface stubs (no local draft model / KV / CUDA graphs).
    # ------------------------------------------------------------------
    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        logger.info("Remote DFlash: draft-side CUDA graphs are remote; no-op.")

    def capture(self, attn_states=None) -> None:
        pass

    def update_finished(self, finished_req_ids: set[str]) -> None:
        if self.tp_rank != 0:
            return
        if finished_req_ids:
            self._pending_finished.update(finished_req_ids)

    # ------------------------------------------------------------------
    # NIXL plumbing
    # ------------------------------------------------------------------
    def _connect(self) -> bool:
        if self._peer is not None:
            return True
        try:
            from nixl._api import nixl_agent, nixl_agent_config

            if self._agent is None:
                # Created once; survives reconnects (local registrations
                # stay valid — only the peer session is replaced).
                agent = nixl_agent(
                    f"dflash-target-{self.port}",
                    nixl_agent_config(backends=["UCX"]),
                )
                agent.register_memory([self.draft_tokens, self.staging])
                self._agent = agent
            agent = self._agent

            conn = socket.create_connection(
                (self.host, self.port), timeout=_CONNECT_TIMEOUT_S
            )
            ids_descs = agent.get_xfer_descs(
                [(self.draft_tokens.data_ptr(),
                  self.draft_tokens.numel() * 8, 0)], "VRAM"
            )
            _send_msg(conn, {
                "meta": agent.get_agent_metadata(),
                "ids_descs": agent.get_serialized_descs(ids_descs),
                "hello": {
                    "k": self.num_speculative_steps,
                    "aux_width": self.aux_width,
                    "dtype": str(self.dtype).split(".")[-1],
                },
            })
            reply = _recv_msg(conn)
            conn.close()

            peer = agent.add_remote_agent(reply["meta"])
            self._peer = peer.decode() if isinstance(peer, bytes) else peer
            ring = agent.deserialize_descs(reply["ring_descs"])
            self._ring_base = ring[0][0]
            self._ring_tokens = reply["ring_tokens"]
            spec = reply["spec"]
            if spec["aux_width"] != self.aux_width:
                raise ValueError(
                    f"drafter service aux_width {spec['aux_width']} != "
                    f"expected {self.aux_width}"
                )
            if spec["num_spec_tokens"] != self.num_speculative_steps:
                raise ValueError(
                    f"drafter block supports k={spec['num_spec_tokens']} but "
                    f"num_speculative_tokens={self.num_speculative_steps}"
                )
            self._row_off = 0
            logger.info("Remote DFlash connected: spec=%s ring_tokens=%d",
                        spec, self._ring_tokens)
            return True
        except Exception as e:
            logger.warning("Remote DFlash connect failed: %s", e)
            self._peer = None
            return False

    def _disconnect(self) -> None:
        """Drop the peer session (keep the agent + local registrations) so
        the next attempt re-bootstraps — the path back from a restarted
        drafter service, and from transient failures (the service's
        persistent listener re-handshakes in place)."""
        peer, self._peer = self._peer, None
        self._ring_tokens = 0
        self._row_off = 0
        if self._agent is not None and peer is not None:
            try:
                self._agent.remove_remote_agent(peer)
            except Exception:
                pass

    def _post_step(self, header: dict, num_tokens: int) -> int:
        import msgpack

        agent = self._agent
        assert agent is not None
        if self._row_off + num_tokens > self._ring_tokens:
            self._row_off = 0
        header["step_id"] = self._step_id
        header["row_off"] = self._row_off
        header["num_tokens"] = num_tokens
        nbytes = num_tokens * self._aux_row_bytes
        local = agent.get_xfer_descs(
            [(self.staging.data_ptr(), nbytes, 0)], "VRAM"
        )
        remote = agent.get_xfer_descs(
            [(self._ring_base + self._row_off * self._aux_row_bytes, nbytes, 0)],
            "VRAM",
        )
        h = agent.initialize_xfer(
            "WRITE", local, remote, self._peer,
            msgpack.packb(header, use_bin_type=True),
        )
        agent.transfer(h)
        deadline = time.monotonic() + self.timeout_s
        while True:
            st = agent.check_xfer_state(h)
            if st == "DONE":
                break
            if st == "ERR" or time.monotonic() > deadline:
                agent.release_xfer_handle(h)
                raise TimeoutError("aux WRITE did not complete")
        agent.release_xfer_handle(h)
        self._row_off += num_tokens
        sid = self._step_id
        self._step_id += 1
        return sid

    def _wait_ack(self, step_id: int) -> bool:
        agent = self._agent
        assert agent is not None
        want = f"ack:{step_id}".encode()
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            for _, msgs in agent.get_new_notifs().items():
                for m in msgs:
                    if m == want:
                        return True
        return False

    def _fail_step(self, num_reqs: int) -> torch.Tensor:
        self._steps_failed += 1
        self._consecutive_failures += 1
        if self._consecutive_failures >= _BREAKER_THRESHOLD:
            self._breaker_open_until = time.monotonic() + _BREAKER_RETRY_S
            # Drop the session: when the breaker closes, _connect() runs a
            # fresh TCP bootstrap (the service may have restarted; its
            # listener is persistent either way).
            self._disconnect()
            logger.warning(
                "Remote DFlash circuit breaker OPEN for %.0fs after %d "
                "failures (session dropped; will re-bootstrap).",
                _BREAKER_RETRY_S, self._consecutive_failures,
            )
        self.draft_tokens[:num_reqs].zero_()
        return self.draft_tokens[:num_reqs]

    # ------------------------------------------------------------------
    # The hot path
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        num_reqs = input_batch.num_reqs
        if dummy_run or is_profile:
            # Profiling / DP-padding path: no transfer, no service traffic,
            # no broadcast (every rank takes this branch together).
            return self.draft_tokens[:num_reqs]

        if self.tp_rank == 0:
            self._propose_rank0(
                input_batch,
                aux_hidden_states,
                num_sampled,
                num_rejected,
                last_sampled,
                next_prefill_tokens,
            )
        if self.tp_size > 1 and num_reqs > 0:
            # Ranks >0 wait here while rank 0 round-trips the service; on
            # any rank-0 failure the broadcast still runs (zeroed drafts).
            self.tp_group.broadcast(self.draft_tokens[:num_reqs], src=0)
        return self.draft_tokens[:num_reqs]

    def _propose_rank0(
        self,
        input_batch: InputBatch,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
    ) -> None:
        """Fill ``self.draft_tokens[:num_reqs]`` via the drafter service
        (zeros on any failure)."""
        num_reqs = input_batch.num_reqs
        if time.monotonic() < self._breaker_open_until:
            self.draft_tokens[:num_reqs].zero_()
            return

        if not self._connect():
            self._fail_step(num_reqs)
            return

        t0 = time.perf_counter()
        num_tokens = input_batch.num_tokens
        assert aux_hidden_states, "DFlash requires aux hidden states"

        # GPU-side gathers, then one D2H copy of the small ints.
        idx = input_batch.idx_mapping
        bonus = torch.where(
            num_sampled > 0,
            last_sampled.view(-1)[idx].to(torch.int64),
            next_prefill_tokens.view(-1)[idx].to(torch.int64),
        )
        self._scratch_gpu[0, :num_reqs] = num_sampled
        self._scratch_gpu[1, :num_reqs] = num_rejected
        self._scratch_gpu[2, :num_reqs] = bonus
        self.staging[:num_tokens].copy_(
            torch.cat(aux_hidden_states, dim=-1)[:num_tokens]
        )
        self._scratch_cpu[:, :num_reqs].copy_(
            self._scratch_gpu[:, :num_reqs], non_blocking=True
        )
        self._copy_event.record()
        self._copy_event.synchronize()

        ns = self._scratch_cpu[0, :num_reqs].tolist()
        nr = self._scratch_cpu[1, :num_reqs].tolist()
        bt = self._scratch_cpu[2, :num_reqs].tolist()
        n_sched = input_batch.num_scheduled_tokens
        start = input_batch.num_computed_tokens_np
        prefill = input_batch.is_prefilling_np

        # EAGLE3 pairs position p with token_{p+1} (the EAGLE shift), so the
        # service needs each request's scheduled token ids, not just the
        # bonus. ~4 bytes/token of header next to 21-43 KB/token of aux rows.
        ids_np = None
        qsl = None
        if self.method == "eagle3":
            ids_np = input_batch.input_ids[:num_tokens].cpu().numpy()
            qsl = input_batch.query_start_loc_np

        reqs = [
            {
                "id": rid,
                "start_pos": int(start[i]),
                "n_sched": int(n_sched[i]),
                "n_rej": int(nr[i]),
                "n_samp": int(ns[i]),
                "bonus": int(bt[i]),
                "prefill": bool(prefill[i]),
                **(
                    {"ids": ids_np[qsl[i]: qsl[i + 1]].tolist()}
                    if ids_np is not None
                    else {}
                ),
            }
            for i, rid in enumerate(input_batch.req_ids)
        ]
        finished = list(self._pending_finished)
        self._pending_finished.clear()

        try:
            sid = self._post_step(
                {"kind": "step", "reqs": reqs, "finished": finished}, num_tokens
            )
            if not self._wait_ack(sid):
                raise TimeoutError(f"no ack for step {sid}")
        except Exception as e:
            logger.warning("Remote DFlash step failed: %s", e)
            # Re-queue the finished ids so the service still frees them.
            self._pending_finished.update(finished)
            self._fail_step(num_reqs)
            return

        self._consecutive_failures = 0
        self._steps_ok += 1
        self._rpc_ms_sum += (time.perf_counter() - t0) * 1e3
        if self._steps_ok % 500 == 0:
            logger.info(
                "Remote DFlash: %d steps ok (%d failed), avg round-trip %.2f ms",
                self._steps_ok, self._steps_failed,
                self._rpc_ms_sum / self._steps_ok,
            )
