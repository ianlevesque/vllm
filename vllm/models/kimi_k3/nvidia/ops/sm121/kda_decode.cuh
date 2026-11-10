// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Fused Kimi-K3 KDA decode for SM12x (GB10 / DGX Spark).
//
// One launch replaces the fallback chain used wherever the SM100 fused KDA
// kernels are unavailable (FlashInfer fused_kda_decode / packed_fused_kda_decode
// are SM10x-only, and the native csrc kernel is restricted to 12/24/48/96 local
// heads and non-speculative rows):
//
//   causal_conv1d_update (+SiLU)  ->  recurrent KDA (L2-normalised q/k, bounded
//   gate decay, delta-rule update)  ->  gated RMSNorm (sigmoid output gate)
//
// for any local head count and for both plain decode (one row per sequence)
// and speculative verification (T <= 8 rows per sequence with per-row state
// slots and an accepted-token rollback offset).
//
// Layout.  A head's 128x128 FP32 state is split into eight 16-row chunks, one
// 128-thread block each, and the eight blocks of a head form one thread-block
// cluster, so the six local heads of a TP16 rank occupy all 48 SMs of a GB10
// (a single block per head is ~7x slower: the kernel is bound by instructions
// and stores per SM, not by memory bandwidth).  Lane l of a warp owns state
// columns [4l, 4l + 4) of that warp's four rows, held in registers across the
// rows of the sequence, so every warp-wide state store writes one contiguous
// 512-byte row: a speculative step has to checkpoint the whole state after
// every row, and those stores (2.3 MB at six heads and six rows, against a
// ~200 GB/s store limit) are what bounds it.  Thread t of each block also
// evaluates q and k convolution channel t of the head, their norms and the
// decay, which the block shares through shared memory (recomputing them per
// block is cheap and avoids a barrier per row); lanes 0..3 of a warp own that
// warp's four v channels.  After the last row the cluster meets once; chunk 0
// then commits the q/k convolution state (the other chunks have read it by
// then) and applies the gated RMSNorm over the head's 128 outputs.
//
// The arithmetic mirrors the Triton reference kernels step for step, including
// the BF16 rounding points of the intermediate tensors they materialise
// (post-convolution q/k/v and the pre-norm output).

#pragma once

#include <cooperative_groups.h>

#include "sm121_common.cuh"

namespace k31sm121 {

constexpr int kKdaDim = 128;            // head dim (K == V)
constexpr int kKdaThreads = 128;
constexpr int kKdaWarps = kKdaThreads / 32;
constexpr int kKdaChunkRows = 16;       // state rows per block
constexpr int kKdaChunks = kKdaDim / kKdaChunkRows;       // blocks per head (one cluster)
constexpr int kKdaRowsPerWarp = kKdaChunkRows / kKdaWarps;    // 4
constexpr int kKdaColsPerLane = kKdaDim / 32;                 // 4
constexpr int kKdaMaxRows = 8;          // rows per sequence in one launch
constexpr int kKdaConvWidth = 4;

struct KdaDecodeArgs {
  const __nv_bfloat16* x;        // [rows, >= 3*LP] projected q|k|v, head-major
  int64_t x_row_stride;
  const float* conv_w;           // [3*LP, 4] channel-major taps
  __nv_bfloat16* conv_state;     // slot/channel/tap strided
  int64_t cs_slot_stride;
  int64_t cs_chan_stride;
  int64_t cs_tap_stride;
  const __nv_bfloat16* raw_g;    // [rows, H, 128] raw decay gate
  int64_t g_row_stride;
  const __nv_bfloat16* raw_beta; // [rows, H] raw beta logits
  int64_t beta_row_stride;
  const float* a_log;            // [H]
  const float* dt_bias;          // [H * 128]
  float* state;                  // [slots, H, 128, 128]
  int64_t state_slot_stride;
  const int32_t* conv_indices;   // [N] conv-state slot per sequence
  int64_t conv_indices_stride;
  const int32_t* state_indices;  // [N, >= T] recurrent-state slot per row
  int64_t state_indices_stride;
  const int32_t* num_accepted;   // [N] or nullptr (plain decode)
  const int32_t* cu_seqlens;     // [N + 1] or nullptr (one row per sequence)
  const __nv_bfloat16* out_gate; // [rows, H, 128] output gate logits
  int64_t og_row_stride;
  const float* norm_w;           // [128]
  __nv_bfloat16* out;            // [rows, H, 128]
  int64_t out_row_stride;
  int H;
  int LP;                        // local projection size == H * 128
  float lower_bound;
  float scale;
  float norm_eps;
};

// One convolution channel: three history taps, four weights, and the two taps
// that survive into the rolled window.
struct KdaConvChannel {
  float c0, c1, c2;
  float w0, w1, w2, w3;
  __nv_bfloat16 keep1, keep2;
};

__device__ __forceinline__ KdaConvChannel kda_conv_open(
    const __nv_bfloat16* cs, int64_t tap_stride, int offset, const float* w) {
  KdaConvChannel ch;
  const __nv_bfloat16 t0 = cs[(offset + 0) * tap_stride];
  ch.keep1 = cs[(offset + 1) * tap_stride];
  ch.keep2 = cs[(offset + 2) * tap_stride];
  ch.c0 = __bfloat162float(t0);
  ch.c1 = __bfloat162float(ch.keep1);
  ch.c2 = __bfloat162float(ch.keep2);
  ch.w0 = w[0];
  ch.w1 = w[1];
  ch.w2 = w[2];
  ch.w3 = w[3];
  return ch;
}

// SiLU(conv) rounded to BF16, as the reference stores it before the recurrence.
__device__ __forceinline__ float kda_conv_step(KdaConvChannel& ch, float xv) {
  float acc = ch.c0 * ch.w0;
  acc += ch.c1 * ch.w1;
  acc += ch.c2 * ch.w2;
  acc += xv * ch.w3;
  ch.c0 = ch.c1;
  ch.c1 = ch.c2;
  ch.c2 = xv;
  return round_bf16(acc / (1.0f + __expf(-acc)));
}

__global__ __cluster_dims__(kKdaChunks, 1, 1) __launch_bounds__(kKdaThreads)
void kda_fused_decode_kernel(const KdaDecodeArgs a) {
  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int group = blockIdx.x / kKdaChunks;   // sequence * H + head
  const int chunk = blockIdx.x - group * kKdaChunks;
  const int n = group / a.H;
  const int h = group - n * a.H;

  // Every block of the cluster takes the same early exits (they depend on the
  // sequence only), so the cluster barrier below is never left short.
  const int bos = a.cu_seqlens == nullptr ? n : a.cu_seqlens[n];
  const int eos = a.cu_seqlens == nullptr ? n + 1 : a.cu_seqlens[n + 1];
  int rows = eos - bos;
  if (rows <= 0) {
    return;
  }
  rows = rows > kKdaMaxRows ? kKdaMaxRows : rows;

  const int offset = a.num_accepted == nullptr ? 0 : a.num_accepted[n] - 1;
  const int64_t conv_slot = a.conv_indices[n * a.conv_indices_stride];
  const int64_t init_slot =
      a.state_indices[n * a.state_indices_stride + offset];

  // Null cache line: the references emit zeros and leave the caches alone.
  if (conv_slot <= 0 || init_slot <= 0) {
    if (chunk == 0) {
      for (int t = 0; t < rows; ++t) {
        a.out[(bos + t) * a.out_row_stride + h * kKdaDim + tid] =
            __float2bfloat16(0.0f);
      }
    }
    return;
  }

  __shared__ float s_q[kKdaDim];
  __shared__ float s_k[kKdaDim];
  __shared__ float s_decay[kKdaDim];
  __shared__ float s_v[kKdaChunkRows];
  __shared__ float s_red[2 * kKdaWarps];
  __shared__ float s_beta;

  // ---- recurrent state fragment into registers ---------------------------
  // S[r][j]: row (chunk * 16 + warp * 4 + r), column (4 * lane + j).
  const int kbase = lane * kKdaColsPerLane;
  const int vbase = chunk * kKdaChunkRows + warp * kKdaRowsPerWarp;
  const int64_t head_state = (int64_t)h * kKdaDim * kKdaDim;
  const int64_t frag = (int64_t)vbase * kKdaDim + kbase;
  float4 S[kKdaRowsPerWarp];
  {
    const float4* src = reinterpret_cast<const float4*>(
        a.state + init_slot * a.state_slot_stride + head_state + frag);
#pragma unroll
    for (int r = 0; r < kKdaRowsPerWarp; ++r) {
      S[r] = src[r * (kKdaDim / 4)];
    }
  }

  // ---- convolution channels ------------------------------------------------
  // Thread t evaluates q and k channel t of the head; lanes 0..3 of a warp
  // also evaluate (and are the only threads that ever touch) its v channels.
  const bool v_owner = lane < kKdaRowsPerWarp;
  const int vrow = vbase + (v_owner ? lane : 0);
  const int64_t ch_q = (int64_t)h * kKdaDim + tid;
  const int64_t ch_k = (int64_t)a.LP + ch_q;
  const int64_t ch_v = 2 * (int64_t)a.LP + (int64_t)h * kKdaDim + vrow;
  __nv_bfloat16* const cs_base = a.conv_state + conv_slot * a.cs_slot_stride;
  __nv_bfloat16* const cs_q = cs_base + ch_q * a.cs_chan_stride;
  __nv_bfloat16* const cs_k = cs_base + ch_k * a.cs_chan_stride;
  __nv_bfloat16* const cs_v = cs_base + ch_v * a.cs_chan_stride;
  KdaConvChannel conv_q = kda_conv_open(cs_q, a.cs_tap_stride, offset,
                                        a.conv_w + ch_q * kKdaConvWidth);
  KdaConvChannel conv_k = kda_conv_open(cs_k, a.cs_tap_stride, offset,
                                        a.conv_w + ch_k * kKdaConvWidth);
  KdaConvChannel conv_v = conv_q;
  if (v_owner) {
    conv_v = kda_conv_open(cs_v, a.cs_tap_stride, offset,
                           a.conv_w + ch_v * kKdaConvWidth);
    // Rolled window: [history2, history3, x_0, ..., x_{rows-1}].
    cs_v[0] = conv_v.keep1;
    cs_v[a.cs_tap_stride] = conv_v.keep2;
  }

  const float exp_a = __expf(a.a_log[h]);
  const float dt_bias = a.dt_bias[h * kKdaDim + tid];

  for (int t = 0; t < rows; ++t) {
    const int64_t row = bos + t;
    const __nv_bfloat16* const x_row = a.x + row * a.x_row_stride;

    // (A) causal convolution + SiLU, decay gate, beta.
    const float qv = kda_conv_step(conv_q, __bfloat162float(x_row[ch_q]));
    const float kv = kda_conv_step(conv_k, __bfloat162float(x_row[ch_k]));
    if (v_owner) {
      const __nv_bfloat16 xb = x_row[ch_v];
      s_v[warp * kKdaRowsPerWarp + lane] =
          kda_conv_step(conv_v, __bfloat162float(xb));
      cs_v[(2 + t) * a.cs_tap_stride] = xb;
    }
    {
      const float gate =
          __bfloat162float(a.raw_g[row * a.g_row_stride + h * kKdaDim + tid]) +
          dt_bias;
      s_decay[tid] = __expf(a.lower_bound * sigmoid_f(exp_a * gate));
    }
    if (tid == 0) {
      s_beta =
          sigmoid_f(__bfloat162float(a.raw_beta[row * a.beta_row_stride + h]));
    }

    // (B) L2-normalise q and k; q carries the 1/sqrt(K) scale.
    {
      float sq = qv * qv;
      float sk = kv * kv;
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        sq += __shfl_xor_sync(0xffffffffu, sq, o);
        sk += __shfl_xor_sync(0xffffffffu, sk, o);
      }
      if (lane == 0) {
        s_red[warp] = sq;
        s_red[kKdaWarps + warp] = sk;
      }
    }
    __syncthreads();
    {
      float sq = 0.0f;
      float sk = 0.0f;
#pragma unroll
      for (int w = 0; w < kKdaWarps; ++w) {
        sq += s_red[w];
        sk += s_red[kKdaWarps + w];
      }
      s_q[tid] = (qv / sqrtf(sq + 1e-6f)) * a.scale;
      s_k[tid] = kv / sqrtf(sk + 1e-6f);
    }
    __syncthreads();

    // (C) delta-rule recurrence on the register-resident state: four rows per
    // warp, their reductions advancing together.
    {
      const float4 dk = *reinterpret_cast<const float4*>(s_decay + kbase);
      const float4 kk = *reinterpret_cast<const float4*>(s_k + kbase);
      const float4 qq = *reinterpret_cast<const float4*>(s_q + kbase);
      float dot[kKdaRowsPerWarp];
#pragma unroll
      for (int r = 0; r < kKdaRowsPerWarp; ++r) {
        S[r].x *= dk.x;
        S[r].y *= dk.y;
        S[r].z *= dk.z;
        S[r].w *= dk.w;
        dot[r] = S[r].x * kk.x + S[r].y * kk.y + S[r].z * kk.z + S[r].w * kk.w;
      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
#pragma unroll
        for (int r = 0; r < kKdaRowsPerWarp; ++r) {
          dot[r] += __shfl_xor_sync(0xffffffffu, dot[r], o);
        }
      }
      const float beta = s_beta;
      float out[kKdaRowsPerWarp];
#pragma unroll
      for (int r = 0; r < kKdaRowsPerWarp; ++r) {
        const float vnew = (s_v[warp * kKdaRowsPerWarp + r] - dot[r]) * beta;
        S[r].x += vnew * kk.x;
        S[r].y += vnew * kk.y;
        S[r].z += vnew * kk.z;
        S[r].w += vnew * kk.w;
        out[r] = S[r].x * qq.x + S[r].y * qq.y + S[r].z * qq.z + S[r].w * qq.w;
      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
#pragma unroll
        for (int r = 0; r < kKdaRowsPerWarp; ++r) {
          out[r] += __shfl_xor_sync(0xffffffffu, out[r], o);
        }
      }
      if (v_owner) {
        // Pre-norm output, BF16 as the reference materialises it.
        float mine = out[0];
#pragma unroll
        for (int r = 1; r < kKdaRowsPerWarp; ++r) {
          mine = lane == r ? out[r] : mine;
        }
        a.out[row * a.out_row_stride + h * kKdaDim + vrow] =
            __float2bfloat16(mine);
      }
      const int64_t dst_slot = a.state_indices[n * a.state_indices_stride + t];
      if (dst_slot > 0) {
        float4* dst = reinterpret_cast<float4*>(
            a.state + dst_slot * a.state_slot_stride + head_state + frag);
#pragma unroll
        for (int r = 0; r < kKdaRowsPerWarp; ++r) {
          dst[r * (kKdaDim / 4)] = S[r];
        }
      }
    }
    // The next row overwrites s_v/s_decay/s_beta and s_red.
    __syncthreads();
  }

  // ---- head barrier: every chunk has read the q/k taps and stored its
  // pre-norm outputs once the cluster met.
  cluster.sync();
  if (chunk != 0) {
    return;
  }

  // Commit the rolled q/k convolution windows.
  cs_q[0] = conv_q.keep1;
  cs_q[a.cs_tap_stride] = conv_q.keep2;
  cs_k[0] = conv_k.keep1;
  cs_k[a.cs_tap_stride] = conv_k.keep2;
  for (int t = 0; t < rows; ++t) {
    const __nv_bfloat16* const x_row = a.x + (bos + t) * a.x_row_stride;
    cs_q[(2 + t) * a.cs_tap_stride] = x_row[ch_q];
    cs_k[(2 + t) * a.cs_tap_stride] = x_row[ch_k];
  }

  // Gated RMSNorm of the head's 128 outputs, row by row.
  const float norm_w = a.norm_w[tid];
  for (int t = 0; t < rows; ++t) {
    const int64_t row = bos + t;
    __nv_bfloat16* const o_ptr =
        a.out + row * a.out_row_stride + h * kKdaDim + tid;
    const float ov = __bfloat162float(*o_ptr);
    const float part = warp_sum(ov * ov);
    if (lane == 0) {
      s_red[warp] = part;
    }
    __syncthreads();
    float ssq = 0.0f;
#pragma unroll
    for (int w = 0; w < kKdaWarps; ++w) {
      ssq += s_red[w];
    }
    const float rstd = 1.0f / sqrtf(ssq / (float)kKdaDim + a.norm_eps);
    const float gate = __bfloat162float(
        a.out_gate[row * a.og_row_stride + h * kKdaDim + tid]);
    *o_ptr = __float2bfloat16(((ov * rstd) * norm_w) * sigmoid_f(gate));
    __syncthreads();
  }
}

inline void launch_kda_fused_decode(const KdaDecodeArgs& args, int num_seqs,
                                    cudaStream_t stream) {
  const dim3 grid(static_cast<unsigned>(num_seqs * args.H * kKdaChunks), 1, 1);
  const dim3 block(kKdaThreads, 1, 1);
  kda_fused_decode_kernel<<<grid, block, 0, stream>>>(args);
}

}  // namespace k31sm121
