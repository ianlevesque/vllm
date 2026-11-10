// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Kimi-K3 AttnRes forward for SM12x (GB10 / DGX Spark).
//
// The production native kernel (csrc/kimi_k3/attn_res_kernel.cu) is SM100-only:
// it caches the FP32 residual rows in TMEM (tcgen05) between its two passes and
// uses the f32x2 packed instructions.  SM12x has neither.  Here each token row
// is one thread-block cluster whose blocks split the row's 16-byte BF16 packs;
// every thread owns one pack of every vector and keeps all the sources' packs
// in registers between the two passes (what TMEM does on SM100).  The blocks
// meet at two cluster barriers: one to combine the per-source statistics of
// the softmax, one for the output RMSNorm.
//
// Sizing follows what a GB10 measures: a row on a single SM is bound by that
// SM's load rate (~27 GB/s) and instruction throughput, a cluster barrier costs
// ~0.23 us where a barrier built from GPU-scope atomics costs ~1.9 us, and a
// warp's global loads and shuffle reductions are latency-bound, so the loads
// of all sources are issued before any is consumed and the reductions of the
// 18 statistics advance together.
//
// Same contract as ops/attn_res.attn_res:
//
//   prefix (+= delta, rounded to BF16, stored)  ->  optional block write  ->
//   softmax over {saved blocks, prefix} of  <rmsnorm(source), qk>  ->
//   mixture  ->  optional output RMSNorm
//
// The cross-block sums use a fixed reduction order, so every block of a row
// (and every replay) derives bit-identical softmax weights and norm factors.

#pragma once

#include <cooperative_groups.h>

#include "sm121_common.cuh"

namespace k31sm121 {

constexpr int kAttnResMaxSources = 9;        // up to 8 saved blocks + the prefix
constexpr int kAttnResSums = 2 * kAttnResMaxSources;
constexpr int kAttnResOutSsq = kAttnResSums; // partial slot of the output norm
constexpr int kAttnResPartials = 20;         // floats per (row, block)
constexpr int kAttnResBlocks = 7;            // blocks per row: one cluster
constexpr int kAttnResThreads = 128;         // 7 * 128 threads * 8 = 7168
constexpr int kAttnResWarps = kAttnResThreads / 32;
constexpr int kAttnResHidden = kAttnResBlocks * kAttnResThreads * 8;

struct AttnResArgs {
  __nv_bfloat16* prefix;            // [rows, hidden], updated in place with delta
  int64_t prefix_row_stride;
  const __nv_bfloat16* delta;       // [rows, hidden] or nullptr
  int64_t delta_row_stride;
  __nv_bfloat16* blocks;            // [rows, R, hidden]
  int64_t block_row_stride;
  int64_t block_slot_stride;
  const __nv_bfloat16* norm_weight;        // [hidden]
  const __nv_bfloat16* qk_weight;          // [hidden]
  const __nv_bfloat16* output_norm_weight; // [hidden] or nullptr
  __nv_bfloat16* output;            // [rows, hidden]
  int64_t output_row_stride;
  float* partial;                   // [rows, 7, 20] cross-block partial sums
  int num_blocks;                   // saved blocks taking part (0..8)
  int block_write_idx;              // -1: no block write
  float eps;
  float output_norm_eps;
};

// Reduce `count` per-thread values over the block: all warp reductions advance
// one shuffle level at a time (their latencies overlap), lane 0 of each warp
// parks its sums, and after one block barrier every thread reads the totals.
template <int kCount>
__device__ __forceinline__ void attn_res_block_sums(float* values,
                                                    float (*scratch)[kCount]) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
    for (int j = 0; j < kCount; ++j) {
      values[j] += __shfl_xor_sync(0xffffffffu, values[j], offset);
    }
  }
  if (lane == 0) {
#pragma unroll
    for (int j = 0; j < kCount; ++j) {
      scratch[warp][j] = values[j];
    }
  }
  __syncthreads();
#pragma unroll
  for (int j = 0; j < kCount; ++j) {
    float total = scratch[0][j];
#pragma unroll
    for (int w = 1; w < kAttnResWarps; ++w) {
      total += scratch[w][j];
    }
    values[j] = total;
  }
}

__global__ __cluster_dims__(kAttnResBlocks, 1, 1)
__launch_bounds__(kAttnResThreads) void attn_res_kernel(const AttnResArgs a) {
  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();
  const int tid = threadIdx.x;
  const int64_t row = blockIdx.x / kAttnResBlocks;
  const int blk = blockIdx.x - (int)row * kAttnResBlocks;
  const int64_t elem = ((int64_t)blk * kAttnResThreads + tid) * 8;
  const int sources = a.num_blocks + 1;

  __shared__ float s_sums[kAttnResWarps][kAttnResSums];
  __shared__ float s_one[kAttnResWarps][1];
  __shared__ float s_total[kAttnResSums];
  __shared__ float s_logit[kAttnResMaxSources];

  float* const row_partial =
      a.partial + row * kAttnResBlocks * kAttnResPartials;
  float* const own_partial = row_partial + blk * kAttnResPartials;

  __nv_bfloat16* const prefix_ptr = a.prefix + row * a.prefix_row_stride + elem;
  __nv_bfloat16* const block_row = a.blocks + row * a.block_row_stride + elem;

  // ---- issue every load before consuming any -------------------------------
  Bf16Pack packs[kAttnResMaxSources];
  Bf16Pack prefix_pack = load_pack(prefix_ptr);
  Bf16Pack delta_pack = prefix_pack;
  Bf16Pack nw_pack = prefix_pack;
  Bf16Pack qw_pack = prefix_pack;
  if (a.delta != nullptr) {
    delta_pack = load_pack(a.delta + row * a.delta_row_stride + elem);
  }
  if (a.num_blocks > 0) {
    nw_pack = load_pack(a.norm_weight + elem);
    qw_pack = load_pack(a.qk_weight + elem);
  }
#pragma unroll
  for (int s = 0; s < kAttnResMaxSources - 1; ++s) {
    packs[s] = prefix_pack;
    // A block that is written below takes the updated prefix instead.
    if (s < a.num_blocks && s != a.block_write_idx) {
      packs[s] = load_pack(block_row + s * a.block_slot_stride);
    }
  }

  // ---- prefix (+ delta), stored back as the BF16 residual source ----------
  if (a.delta != nullptr) {
    float pv[8];
    float dv[8];
    unpack_bf16(prefix_pack, pv);
    unpack_bf16(delta_pack, dv);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      pv[i] += dv[i];
    }
    prefix_pack = pack_bf16(pv);
    store_pack(prefix_ptr, prefix_pack);
  }
  if (a.block_write_idx >= 0) {
    store_pack(block_row + a.block_write_idx * a.block_slot_stride, prefix_pack);
  }
#pragma unroll
  for (int s = 0; s < kAttnResMaxSources; ++s) {
    // Source num_blocks is the prefix; so is a saved block written just now.
    if (s == a.num_blocks || (s < a.num_blocks && s == a.block_write_idx)) {
      packs[s] = prefix_pack;
    }
  }

  float mixed[8];
  if (a.num_blocks == 0) {
    // Only the prefix takes part: its softmax weight is exactly one.
    unpack_bf16(prefix_pack, mixed);
  } else {
    // ---- pass 1: per-source sum of squares and <source, qk> --------------
    float qk[8];
    {
      float nw[8];
      float qw[8];
      unpack_bf16(nw_pack, nw);
      unpack_bf16(qw_pack, qw);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        qk[i] = nw[i] * qw[i];
      }
    }
    float sums[kAttnResSums];
#pragma unroll
    for (int s = 0; s < kAttnResMaxSources; ++s) {
      float ssq = 0.0f;
      float dot = 0.0f;
      if (s < sources) {
        float v[8];
        unpack_bf16(packs[s], v);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          ssq += v[i] * v[i];
          dot += v[i] * qk[i];
        }
      }
      sums[2 * s] = ssq;
      sums[2 * s + 1] = dot;
    }
    attn_res_block_sums<kAttnResSums>(sums, s_sums);
    if (tid < kAttnResSums) {
      own_partial[tid] = sums[tid];
    }
    cluster.sync();

    // Row totals in block order (identical in every block).  Only a few
    // threads touch global memory here: a load is ~60 ns of latency, so each
    // statistic is summed by one thread and shared through shared memory.
    if (tid < kAttnResSums) {
      float total = row_partial[tid];
#pragma unroll
      for (int b = 1; b < kAttnResBlocks; ++b) {
        total += row_partial[b * kAttnResPartials + tid];
      }
      s_total[tid] = total;
    }
    __syncthreads();
    if (tid < kAttnResMaxSources) {
      const float inv_hidden = 1.0f / (float)kAttnResHidden;
      s_logit[tid] =
          tid < sources
              ? s_total[2 * tid + 1] *
                    rsqrtf(s_total[2 * tid] * inv_hidden + a.eps)
              : -3.402823466e+38f;
    }
    __syncthreads();
    float scale[kAttnResMaxSources];
    float denom = 0.0f;
    {
      float max_logit = -3.402823466e+38f;
#pragma unroll
      for (int s = 0; s < kAttnResMaxSources; ++s) {
        if (s < sources) {
          max_logit = fmaxf(max_logit, s_logit[s]);
        }
      }
#pragma unroll
      for (int s = 0; s < kAttnResMaxSources; ++s) {
        scale[s] = 0.0f;
        if (s < sources) {
          scale[s] = __expf(s_logit[s] - max_logit);
          denom += scale[s];
        }
      }
    }

    // ---- pass 2: softmax mixture from the register-resident packs --------
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      mixed[i] = 0.0f;
    }
#pragma unroll
    for (int s = 0; s < kAttnResMaxSources; ++s) {
      if (s < sources) {
        float v[8];
        unpack_bf16(packs[s], v);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          mixed[i] += scale[s] * v[i];
        }
      }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      mixed[i] /= denom;
    }
  }

  // ---- optional output RMSNorm, then store ---------------------------------
  __nv_bfloat16* const out_ptr = a.output + row * a.output_row_stride + elem;
  if (a.output_norm_weight != nullptr) {
    const Bf16Pack ow_pack = load_pack(a.output_norm_weight + elem);
    float ssq[1] = {0.0f};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      ssq[0] += mixed[i] * mixed[i];
    }
    attn_res_block_sums<1>(ssq, s_one);
    if (tid == 0) {
      own_partial[kAttnResOutSsq] = ssq[0];
    }
    cluster.sync();
    if (tid == 0) {
      float total = row_partial[kAttnResOutSsq];
#pragma unroll
      for (int b = 1; b < kAttnResBlocks; ++b) {
        total += row_partial[b * kAttnResPartials + kAttnResOutSsq];
      }
      s_total[0] =
          rsqrtf(total * (1.0f / (float)kAttnResHidden) + a.output_norm_eps);
    }
    __syncthreads();
    const float rstd = s_total[0];
    float ow[8];
    unpack_bf16(ow_pack, ow);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      mixed[i] = (mixed[i] * rstd) * ow[i];
    }
  }
  store_pack(out_ptr, pack_bf16(mixed));
}

inline void launch_attn_res(const AttnResArgs& args, int rows,
                            cudaStream_t stream) {
  const dim3 grid(static_cast<unsigned>(rows * kAttnResBlocks), 1, 1);
  const dim3 block(kAttnResThreads, 1, 1);
  attn_res_kernel<<<grid, block, 0, stream>>>(args);
}

}  // namespace k31sm121
