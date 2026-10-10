# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RoCEnante: adapter for the b12x one-shot RoCE collectives (multi-node DGX Spark TP).

Thin shim: capability voting, construction, and size gating live here; the
protocol lives in ``b12x.comm.roce``.  Enabled with
``VLLM_ENABLE_ROCE_ALLREDUCE=1`` for tensor-parallel groups whose ranks span
nodes; single-node groups keep their existing backends.

Contract with the runtime (``b12x.comm.roce.API_VERSION`` ==
``REQUIRED_B12X_ROCE_API_VERSION``):

- Every rank parses the size limits and checks the API version before the
  vote; the parsed limits are exchanged and must be identical, and the
  runtime itself refuses ranks whose ABI, HCA count, slot geometry, spin
  limit or launch geometry differ.  Any rank that cannot take part disables
  the backend on every rank, at initialization only.
- Dispatch is rank-invariant: eligibility depends on dtype, shape, contiguity
  and size, never on pointer values, so all ranks route the same collective.
- Failures are fail-stop, never a fallback: a wait that times out freezes the
  runtime, later launches do nothing, and ``check_health`` (called by the
  worker after each step's host synchronization) raises so the step's output
  never leaves the worker.  Peers starve on the stalled rank and raise too.
- The runtime orders collectives across streams with an event and requires a
  single stream inside a CUDA graph capture, which is how vLLM captures.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

import vllm.envs as envs
from vllm.distributed.parallel_state import in_the_same_node_as
from vllm.logger import init_logger

logger = init_logger(__name__)


REQUIRED_B12X_ROCE_API_VERSION = 1


def _parse_byte_size(value: str) -> int:
    """Parse the RoCE limits without importing optional PCIe collectives."""
    normalized = value.upper().strip()
    for suffix, multiplier in (
        ("KB", 1024),
        ("MB", 1024**2),
        ("K", 1024),
        ("M", 1024**2),
    ):
        if normalized.endswith(suffix):
            return int(normalized[: -len(suffix)]) * multiplier
    return int(normalized)


class B12xRoceAllReduce:
    """Route eligible tensor-parallel all-reduces to ``b12x.comm.roce``."""

    backend_name = "B12X_ROCENANTE"

    def __init__(
        self,
        group: ProcessGroup,
        device_group: ProcessGroup | None,
        device: torch.device,
    ) -> None:
        self.disabled = True
        self.group = group
        self.device_group = device_group
        self.device = device
        self.rank = dist.get_rank(group=group)
        self.world_size = dist.get_world_size(group=group)
        self._runtime = None
        self._hierarchical_runtimes = []
        self._hierarchical_groups = []
        self._hierarchical_min_size = 0
        self._hierarchical_max_size = 0
        self._announced_hierarchy = False
        self._announced = False
        self._announced_gather = False

        if device_group is None:
            logger.warning("RoCEnante requires a CUDA process group.")
            return
        if all(in_the_same_node_as(group, source_rank=0)):
            logger.info("RoCEnante skipped: group is single-node.")
            return

        # Vote before the collective constructor so a rank that cannot take
        # part (missing package, wrong API version, unsupported device, or an
        # unparsable limit) disables the backend on every rank instead of
        # leaving peers in the runtime's setup exchange.  The parsed limits
        # travel with the vote and must be identical everywhere.
        reason, limits = self._local_capability()
        verdict = self._exchange_vote(reason, limits)
        if verdict is not None:
            logger.warning("RoCEnante disabled on every rank: %s", verdict)
            return
        max_size, max_gather, hierarchical_min = limits

        from b12x.comm import roce

        try:
            # Exchange setup over the CPU (gloo) group: using the torch NCCL
            # group would create a torch NCCL communicator that vLLM otherwise
            # never needs, costing ~3.4 GB of unified memory per rank on Spark.
            self._runtime = roce.AllReduce.from_exchange_group(
                exchange_group=group,
                device=device,
                max_size=max_size,
                max_gather_bytes=max_gather,
            )
        except Exception as exc:  # noqa: BLE001 - the runtime already coordinated ranks
            logger.warning("RoCEnante initialization failed: %s", exc)
            return
        self.disabled = False
        if hierarchical_min:
            # Initialization failure is fatal and coordinated across TP ranks;
            # never leave some ranks using a hierarchy and others using flat.
            self._initialize_hierarchy(max_size, hierarchical_min)
        if self.rank == 0:
            logger.info(
                "Using RoCEnante (b12x one-shot RoCE collectives): world=%d, hcas=%s, "
                "all-reduce max=%d bytes, all-gather shard max=%d bytes.",
                self.world_size,
                ",".join(self._runtime.hca_names),
                max_size,
                max_gather,
            )

    def _local_capability(self) -> tuple[str | None, tuple[int, int, int] | None]:
        """Evaluate this rank's ability to take part, without any collective.

        Returns:
            A pair of the reason this rank cannot take part (None when it can)
            and the parsed ``(max_size, max_gather)`` limits (None on failure).

        """
        try:
            from b12x.comm import roce
        except ImportError as exc:  # missing package or a broken native build
            return f"b12x.comm.roce is not importable: {exc}", None
        api = getattr(roce, "API_VERSION", None)
        if api != REQUIRED_B12X_ROCE_API_VERSION:
            needed = REQUIRED_B12X_ROCE_API_VERSION
            return f"b12x.comm.roce API version {api}, adapter needs {needed}", None
        if not roce.is_supported(self.device):
            return "needs an integrated GPU with an active RDMA device", None
        try:
            limits = (
                _parse_byte_size(envs.VLLM_ROCE_ALLREDUCE_MAX_SIZE),
                _parse_byte_size(envs.VLLM_ROCE_ALLGATHER_MAX_SIZE),
                _parse_byte_size(envs.VLLM_ROCE_HIERARCHICAL_MIN_SIZE)
                if envs.VLLM_ROCE_HIERARCHICAL_ALLREDUCE
                else 0,
            )
            if limits[2] and (
                self.world_size != 16 or not 16 <= limits[2] <= limits[0] // 2
            ):
                return (
                    "RoCE hierarchy requires TP16 and a valid FP32 payload range",
                    None,
                )
        except Exception as exc:  # noqa: BLE001 - reported through the vote
            return f"invalid RoCEnante size limit: {exc}", None
        return None, limits

    def _exchange_vote(
        self, reason: str | None, limits: tuple[int, int, int] | None
    ) -> str | None:
        """Gather every rank's capability result over the CPU group.

        Args:
            reason: This rank's reason for not taking part, or None.
            limits: This rank's parsed ``(max_size, max_gather)``, or None.

        Returns:
            None when every rank can proceed with identical limits, else the
            text naming the ranks that cannot or whose limits differ.

        """
        votes: list[tuple[str | None, tuple[int, int, int] | None]] = [
            (None, None)
        ] * self.world_size
        dist.all_gather_object(votes, (reason, limits), group=self.group)
        failures = [f"rank {i}: {r}" for i, (r, _) in enumerate(votes) if r]
        if failures:
            return "; ".join(failures)
        reference = votes[0][1]
        differing = [
            f"rank {i}: {lim}" for i, (_, lim) in enumerate(votes) if lim != reference
        ]
        if differing:
            return (
                f"size limits differ across ranks (rank 0: {reference}; "
                + "; ".join(differing)
                + ")"
            )
        return None

    def _initialize_hierarchy(self, max_size: int, minimum: int) -> None:
        """Compose existing four-rank RoCE kernels with FP32 intermediates.

        Each row first reduces its four TP ranks; each column then combines
        the four row results. BF16 is converted only at the input/output,
        avoiding an extra BF16 rounding between the two reductions. This is
        opt-in, size-gated, and has no new GPU kernel or transport protocol.
        """
        from b12x.comm import roce

        ranks = dist.get_process_group_ranks(self.group)
        assert len(ranks) == 16
        member_groups = []
        for columns in (False, True):
            for index in range(4):
                offsets = (
                    [row * 4 + index for row in range(4)]
                    if columns
                    else list(range(index * 4, (index + 1) * 4))
                )
                group = dist.new_group(
                    ranks=[ranks[offset] for offset in offsets],
                    backend="gloo",
                    timeout=timedelta(seconds=120),
                    use_local_synchronization=True,
                )
                if self.rank in offsets:
                    member_groups.append(group)
                    self._hierarchical_groups.append(group)
        assert len(member_groups) == 2
        for group in member_groups:
            error = None
            try:
                runtime = roce.AllReduce(
                    exchange_group=group,
                    device=self.device,
                    max_size=max_size,
                    max_gather_bytes=max_size,
                )
                self._hierarchical_runtimes.append(runtime)
                runtime.prepare((torch.float32,))
            except Exception as exc:
                error = str(exc)
            errors = [None] * self.world_size
            dist.all_gather_object(errors, error, group=self.group)
            if any(errors):
                self.close()
                raise RuntimeError(f"RoCE hierarchy initialization failed: {errors}")
        self._hierarchical_min_size = minimum
        # BF16 -> FP32 doubles each subgroup payload. Keep the original
        # large-message/fallback range and other dtypes on their flat path.
        self._hierarchical_max_size = max_size // 2
        if self.rank == 0:
            logger.info(
                "RoCEnante FP32 4x4 hierarchy ready: BF16 payload %d..%d bytes",
                minimum,
                self._hierarchical_max_size,
            )

    def _should_hierarchical(self, inp: torch.Tensor) -> bool:
        return (
            bool(self._hierarchical_runtimes)
            and inp.dtype == torch.bfloat16
            and self._hierarchical_min_size
            <= inp.numel() * inp.element_size()
            <= self._hierarchical_max_size
        )

    def check_health(self) -> None:
        """Fail-stop check of the runtime.

        Raises:
            RuntimeError: When a RoCEnante wait timed out or its proxy died.

        """
        if not self.disabled and self._runtime is not None:
            self._runtime.check_health()
            for runtime in self._hierarchical_runtimes:
                runtime.check_health()

    def should_custom_ar(self, inp: torch.Tensor) -> bool:
        return not self.disabled and self._runtime.should_allreduce(inp)

    def custom_all_reduce(self, inp: torch.Tensor) -> torch.Tensor | None:
        if not self.should_custom_ar(inp):
            return None
        if not self._announced:
            self._announced = True
            # One confirmation line, rank 0 only; workers keep it at debug.
            log = logger.info if self.rank == 0 else logger.debug
            log(
                "RoCEnante all-reduce is live: first routed all-reduce is %d bytes "
                "(%s); NCCL remains the fallback above %s.",
                inp.numel() * inp.element_size(),
                str(inp.dtype).replace("torch.", ""),
                envs.VLLM_ROCE_ALLREDUCE_MAX_SIZE,
            )
        if self._should_hierarchical(inp):
            if not self._announced_hierarchy:
                self._announced_hierarchy = True
                log = logger.info if self.rank == 0 else logger.debug
                log(
                    "RoCEnante FP32 hierarchy is live: first BF16 reduction %d bytes",
                    inp.numel() * inp.element_size(),
                )
            partial = self._hierarchical_runtimes[0].all_reduce(inp.float())
            reduced = self._hierarchical_runtimes[1].all_reduce(partial)
            return reduced.to(inp.dtype)
        return self._runtime.all_reduce(inp)

    def should_all_gather(self, inp: torch.Tensor, dim: int) -> bool:
        return not self.disabled and self._runtime.should_all_gather(inp, dim)

    def all_gather(self, inp: torch.Tensor, dim: int) -> torch.Tensor:
        """Concatenate along ``dim`` (0 or last) directly in the output layout."""
        if not self._announced_gather:
            self._announced_gather = True
            log = logger.info if self.rank == 0 else logger.debug
            log(
                "RoCEnante all-gather is live: first routed shard is %s %s "
                "along dim %d.",
                tuple(inp.shape),
                str(inp.dtype).replace("torch.", ""),
                dim,
            )
        return self._runtime.all_gather(inp, dim=dim)

    def supports_fused_add_rms_norm(self) -> bool:
        return False

    @contextmanager
    def capture(self, stream: torch.cuda.Stream | None = None):
        if self.disabled:
            yield
            return
        # Compile and allocate scratch before capture; the runtime refuses both
        # inside a graph.  vLLM's gather shards have 16-byte rows (direct
        # layout), so the padded-gather scratch is not requested here.
        self._runtime.prepare((torch.bfloat16, torch.float16, torch.float32))
        with ExitStack() as contexts:
            contexts.enter_context(self._runtime.capture(stream=stream))
            for runtime in self._hierarchical_runtimes:
                runtime.prepare((torch.float32,))
                contexts.enter_context(runtime.capture(stream=stream))
            yield

    def close(self) -> None:
        for runtime in self._hierarchical_runtimes:
            runtime.close()
        self._hierarchical_runtimes.clear()
        for group in self._hierarchical_groups:
            dist.destroy_process_group(group)
        self._hierarchical_groups.clear()
        if self._runtime is not None:
            self._runtime.close()
            self._runtime = None
        self.disabled = True
