# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for block-quant MoE loading with a padded TP partition.

A codec block is atomic (256 weights for IQ2_*), so a global intermediate
like 3072 cannot shard evenly over TP16 (192/rank).  maybe_roundup_sizes
rounds the partition up to a whole block; the checkpoint keeps the real
3072 and weight_loader copies only each rank's intersection with it,
leaving exact zero blocks in the padded remainder.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.quantization.modelopt_block_quant import (
    ModelOptBlockQuantMoEMethod,
)

BLOCK = 256
BYTES = 74
HIDDEN = 512
TP = 4
LOCAL_I = 256  # padded partition
REAL_GLOBAL = 768  # unpadded checkpoint intermediate (192/rank)


def make_method(tp_rank: int) -> ModelOptBlockQuantMoEMethod:
    method = object.__new__(ModelOptBlockQuantMoEMethod)
    method.codec = "iq2_xs"
    method.block_size = BLOCK
    method.block_bytes = BYTES
    method.gated = True
    method.moe = SimpleNamespace(
        intermediate_size_per_partition=LOCAL_I,
        hidden_dim=HIDDEN,
        tp_rank=tp_rank,
        moe_parallel_config=SimpleNamespace(tp_size=TP),
    )
    return method


def marked(rows: int) -> torch.Tensor:
    """w13-like tensor whose every byte in row r equals r % 251 + 1."""
    values = (torch.arange(rows) % 251 + 1).to(torch.uint8)
    return values[:, None, None].expand(rows, HIDDEN // BLOCK, BYTES).contiguous()


class TestRoundup:
    def test_rounds_up_to_block(self):
        method = make_method(0)
        assert method.maybe_roundup_sizes(HIDDEN, 192, None, None) == (HIDDEN, 256)
        assert method.maybe_roundup_sizes(HIDDEN, 256, None, None) == (HIDDEN, 256)
        # Ragged FC2 K is supported at 128 granularity (3072/8 at TP8).
        assert method.maybe_roundup_sizes(HIDDEN, 384, None, None) == (HIDDEN, 384)
        assert method.maybe_roundup_sizes(HIDDEN, 128, None, None) == (HIDDEN, 128)

    def test_hidden_must_align(self):
        method = make_method(0)
        with pytest.raises(ValueError, match="hidden size must align"):
            method.maybe_roundup_sizes(1000, 256, None, None)


class TestWeightLoaderUnpadded:
    def load(self, tp_rank, shard_id, loaded, num_experts=2):
        method = make_method(tp_rank)
        if shard_id == "w2":
            param = SimpleNamespace(
                data=torch.zeros(
                    num_experts, HIDDEN, LOCAL_I // BLOCK, BYTES, dtype=torch.uint8
                ),
                shape=(num_experts,),
            )
        else:
            param = SimpleNamespace(
                data=torch.zeros(
                    num_experts, 2 * LOCAL_I, HIDDEN // BLOCK, BYTES,
                    dtype=torch.uint8,
                ),
                shape=(num_experts,),
            )
        method.weight_loader(param, loaded, "w", shard_id, 1)
        return param.data[1]

    def test_w1_full_overlap_rank(self):
        got = self.load(0, "w1", marked(REAL_GLOBAL))
        torch.testing.assert_close(got[:LOCAL_I], marked(256))
        assert int(got[LOCAL_I:].abs().max()) == 0  # w3 half untouched

    def test_w1_last_real_rank(self):
        got = self.load(2, "w1", marked(REAL_GLOBAL))
        torch.testing.assert_close(got[:LOCAL_I], marked(768)[512:])

    def test_w1_rank_entirely_in_padding_stays_zero(self):
        got = self.load(3, "w1", marked(REAL_GLOBAL))
        assert int(got.abs().max()) == 0

    def test_w3_writes_second_half(self):
        got = self.load(1, "w3", marked(REAL_GLOBAL))
        assert int(got[:LOCAL_I].abs().max()) == 0
        torch.testing.assert_close(got[LOCAL_I:], marked(768)[256:512])

    def test_w2_block_columns(self):
        loaded = torch.zeros(HIDDEN, REAL_GLOBAL // BLOCK, BYTES, dtype=torch.uint8)
        for b in range(REAL_GLOBAL // BLOCK):
            loaded[:, b] = b + 1
        got = self.load(2, "w2", loaded)
        assert got.shape == (HIDDEN, 1, BYTES)
        assert int(got.min()) == 3 and int(got.max()) == 3
        got_pad = self.load(3, "w2", loaded)
        assert int(got_pad.abs().max()) == 0

    def test_padded_checkpoint_still_loads_exactly(self):
        got = self.load(3, "w1", marked(LOCAL_I * TP))
        torch.testing.assert_close(got[:LOCAL_I], marked(1024)[768:])

    @pytest.mark.parametrize(
        "loaded",
        [
            marked(LOCAL_I * TP + BLOCK),  # larger than the padded partition
            marked(300),  # global intermediate not block-aligned
            marked(REAL_GLOBAL).to(torch.int16),  # wrong dtype
            torch.zeros(REAL_GLOBAL, BYTES, dtype=torch.uint8),  # not 3D
        ],
    )
    def test_rejects_invalid_tensors(self, loaded):
        method = make_method(0)
        param = SimpleNamespace(
            data=torch.zeros(1, 2 * LOCAL_I, HIDDEN // BLOCK, BYTES,
                             dtype=torch.uint8),
            shape=(1,),
        )
        with pytest.raises(ValueError, match="invalid packed tensor"):
            method.weight_loader(param, loaded, "w", "w1", 0)


SEG_LOCAL_I = 384  # ragged TP8-style partition of a 768 global intermediate
SEG_TP = 2
SEG_BYTES = 66  # iq2_xxs


def make_segmented_method(tp_rank: int) -> ModelOptBlockQuantMoEMethod:
    method = object.__new__(ModelOptBlockQuantMoEMethod)
    method.codec = "iq2_xxs"
    method.block_size = BLOCK
    method.block_bytes = SEG_BYTES
    method.gated = True
    method.moe = SimpleNamespace(
        intermediate_size_per_partition=SEG_LOCAL_I,
        hidden_dim=HIDDEN,
        tp_rank=tp_rank,
        moe_parallel_config=SimpleNamespace(tp_size=SEG_TP),
    )
    return method


class TestWeightLoaderSegmented:
    def load_w2(self, tp_rank, loaded):
        method = make_segmented_method(tp_rank)
        param = SimpleNamespace(
            data=torch.zeros(1, HIDDEN, 2, SEG_BYTES, dtype=torch.uint8),
            shape=(1,),
        )
        method.weight_loader(param, loaded, "w", "w2", 0)
        return param.data[0]

    @staticmethod
    def segmented_w2() -> torch.Tensor:
        loaded = torch.zeros(HIDDEN, 4, SEG_BYTES, dtype=torch.uint8)
        for b in range(4):
            loaded[:, b] = b + 1
        return loaded

    def test_w2_segmented_ranks_copy_own_block_pair(self):
        loaded = self.segmented_w2()
        got0 = self.load_w2(0, loaded)
        assert int(got0[:, 0].min()) == 1 and int(got0[:, 1].min()) == 2
        got1 = self.load_w2(1, loaded)
        assert int(got1[:, 0].min()) == 3 and int(got1[:, 1].min()) == 4

    def test_w2_standard_layout_rejected_for_ragged_partition(self):
        loaded = torch.zeros(HIDDEN, 3, SEG_BYTES, dtype=torch.uint8)
        with pytest.raises(ValueError, match="segmented w2"):
            self.load_w2(1, loaded)

    def test_w1_rows_copy_exactly_at_ragged_partition(self):
        method = make_segmented_method(1)
        param = SimpleNamespace(
            data=torch.zeros(1, 2 * SEG_LOCAL_I, HIDDEN // BLOCK, SEG_BYTES,
                             dtype=torch.uint8),
            shape=(1,),
        )
        values = (torch.arange(768) % 251 + 1).to(torch.uint8)
        loaded = values[:, None, None].expand(
            768, HIDDEN // BLOCK, SEG_BYTES).contiguous()
        method.weight_loader(param, loaded, "w", "w1", 0)
        torch.testing.assert_close(param.data[0, :SEG_LOCAL_I], loaded[384:])
