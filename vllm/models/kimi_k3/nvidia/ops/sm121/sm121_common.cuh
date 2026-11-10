// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Shared device helpers for the Kimi-K3 SM12x fused kernels.
//
// GB10 has 48 SMs and these kernels are bound by instruction throughput per
// SM, not by memory, so each logical unit of work (a KDA head, an AttnRes
// row, a collective) is split over several thread blocks.  The blocks of one
// group meet at explicit barriers built from GPU-scope atomics on a small
// device-resident scratch word pair; SM12x has no TMEM and these kernels do
// not rely on thread-block clusters.

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace k31sm121 {

__device__ __forceinline__ uint32_t ld_acquire_gpu_u32(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];"
               : "=r"(v)
               : "l"(p)
               : "memory");
  return v;
}

__device__ __forceinline__ uint32_t ld_relaxed_gpu_u32(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];"
               : "=r"(v)
               : "l"(p)
               : "memory");
  return v;
}

__device__ __forceinline__ void st_release_gpu_u32(uint32_t* p, uint32_t v) {
  asm volatile("st.release.gpu.global.u32 [%0], %1;" : : "l"(p), "r"(v)
               : "memory");
}

__device__ __forceinline__ uint32_t atom_add_gpu_u32(uint32_t* p, uint32_t v) {
  uint32_t prior;
  asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], %2;"
               : "=r"(prior)
               : "l"(p), "r"(v)
               : "memory");
  return prior;
}

__device__ __forceinline__ void fence_sc_gpu() {
  asm volatile("fence.sc.gpu;" : : : "memory");
}

// ---- cross-block group barrier ---------------------------------------------
//
// `count` blocks share the word pair {arrive, depart}, both zero between
// launches.  Exactly one thread per block calls these.  A block publishes its
// data, arrives, waits until every block of the group arrived, consumes the
// group's data, and departs; the last block to depart re-zeroes both words,
// which is safe because every block is past its wait by then and the next
// launch using the pair cannot begin before this kernel completes.

__device__ __forceinline__ void group_arrive(uint32_t* arrive) {
  fence_sc_gpu();
  atom_add_gpu_u32(arrive, 1u);
}

__device__ __forceinline__ void group_wait(const uint32_t* arrive,
                                           uint32_t count) {
  while (ld_acquire_gpu_u32(arrive) < count) {
  }
}

__device__ __forceinline__ void group_depart(uint32_t* arrive, uint32_t* depart,
                                             uint32_t count) {
  const uint32_t prior = atom_add_gpu_u32(depart, 1u);
  if (prior + 1u == count) {
    st_release_gpu_u32(depart, 0u);
    st_release_gpu_u32(arrive, 0u);
  }
}

// ---- small numeric helpers ---------------------------------------------------

__device__ __forceinline__ float sigmoid_f(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ float round_bf16(float x) {
  return __bfloat162float(__float2bfloat16(x));
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  }
  return value;
}

// One 16-byte pack of eight BF16 values, as four packed pairs.
struct Bf16Pack {
  uint32_t w[4];
};

__device__ __forceinline__ Bf16Pack load_pack(const __nv_bfloat16* p) {
  const uint4 v = *reinterpret_cast<const uint4*>(p);
  return {{v.x, v.y, v.z, v.w}};
}

__device__ __forceinline__ void store_pack(__nv_bfloat16* p, const Bf16Pack& v) {
  *reinterpret_cast<uint4*>(p) = make_uint4(v.w[0], v.w[1], v.w[2], v.w[3]);
}

__device__ __forceinline__ void unpack_bf16(const Bf16Pack& v, float* out) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    __nv_bfloat162 pair;
    *reinterpret_cast<uint32_t*>(&pair) = v.w[i];
    const float2 f = __bfloat1622float2(pair);
    out[2 * i] = f.x;
    out[2 * i + 1] = f.y;
  }
}

__device__ __forceinline__ Bf16Pack pack_bf16(const float* in) {
  Bf16Pack v;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    __nv_bfloat162 pair;
    pair.x = __float2bfloat16(in[2 * i]);
    pair.y = __float2bfloat16(in[2 * i + 1]);
    v.w[i] = *reinterpret_cast<uint32_t*>(&pair);
  }
  return v;
}

}  // namespace k31sm121
