// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Micro-probes used to size the SM12x fused kernels (validation only): launch
// floor, intra-block and cross-block barrier cost, arithmetic throughput per
// SM, shared-memory reads and global load bandwidth, each as a function of the
// grid and block shape.

#pragma once

#include <cooperative_groups.h>

#include "sm121_common.cuh"

namespace k31sm121 {

struct ProbeArgs {
  const float* data;     // global buffer (>= grid * threads * 16 * iters bytes for mode 5)
  float* out;            // [grid] one result per block
  uint32_t* sync;        // [2] barrier words
  int mode;
  int iters;
  int group;             // blocks per cross-block group (mode 2)
};

__global__ void probe_kernel(const ProbeArgs a) {
  const int tid = threadIdx.x;
  __shared__ float s_buf[1024];
  float acc = (float)(tid + 1) * 1e-3f;
  if (a.mode == 0) {
    return;
  } else if (a.mode == 1) {
    for (int i = 0; i < a.iters; ++i) {
      __syncthreads();
    }
  } else if (a.mode == 2) {
    for (int i = 0; i < a.iters; ++i) {
      if (tid == 0) {
        group_arrive(a.sync);
        group_wait(a.sync, (uint32_t)a.group);
        group_depart(a.sync, a.sync + 1, (uint32_t)a.group);
        // Let the last departer's reset land before the next round's arrivals.
        while (ld_acquire_gpu_u32(a.sync + 1) != 0u) {
        }
      }
      __syncthreads();
    }
  } else if (a.mode == 3) {
    // Four independent FMA chains per thread.
    float b = acc + 1.0f, c = acc + 2.0f, d = acc + 3.0f;
    for (int i = 0; i < a.iters; ++i) {
      acc = acc * 0.999f + 0.001f;
      b = b * 0.998f + 0.002f;
      c = c * 0.997f + 0.003f;
      d = d * 0.996f + 0.004f;
    }
    acc += b + c + d;
  } else if (a.mode == 4) {
    s_buf[tid & 1023] = acc;
    __syncthreads();
    for (int i = 0; i < a.iters; ++i) {
      acc += s_buf[(tid + i) & 1023];
    }
  } else if (a.mode == 5) {
    // Each thread streams `iters` 16-byte packs from its own contiguous range.
    const int64_t base = ((int64_t)blockIdx.x * blockDim.x + tid) * a.iters;
    const float4* p = reinterpret_cast<const float4*>(a.data) + base;
    for (int i = 0; i < a.iters; ++i) {
      const float4 v = p[i];
      acc += v.x + v.y + v.z + v.w;
    }
  } else if (a.mode == 6) {
    for (int i = 0; i < a.iters; ++i) {
      acc = __expf(-acc * 0.5f) + 0.25f;
    }
  } else if (a.mode == 7) {
    // Warp shuffle reductions.
    for (int i = 0; i < a.iters; ++i) {
      acc = warp_sum(acc) * (1.0f / 33.0f);
    }
  } else if (a.mode == 8) {
    // Each thread stores `iters` 16-byte packs to its own contiguous range.
    const int64_t base = ((int64_t)blockIdx.x * blockDim.x + tid) * a.iters;
    float4* p = reinterpret_cast<float4*>(const_cast<float*>(a.data)) + base;
    for (int i = 0; i < a.iters; ++i) {
      p[i] = make_float4(acc, acc, acc, acc);
    }
  }
  else if (a.mode == 10) {
    // System-scope 16-byte loads (the reduce path of a RoCE collective).
    const int64_t base = ((int64_t)blockIdx.x * blockDim.x + tid) * a.iters;
    const uint32_t* p = reinterpret_cast<const uint32_t*>(a.data) + base * 4;
    for (int i = 0; i < a.iters; ++i) {
      uint32_t w0, w1, w2, w3;
      asm volatile("ld.relaxed.sys.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                   : "=r"(w0), "=r"(w1), "=r"(w2), "=r"(w3)
                   : "l"(p + 4 * i)
                   : "memory");
      acc += (float)(w0 ^ w1 ^ w2 ^ w3);
    }
  } else if (a.mode == 11) {
    // GPU-scope atomic add, one per iteration (thread 0 of each block).
    if (tid == 0) {
      for (int i = 0; i < a.iters; ++i) {
        atom_add_gpu_u32(a.sync + 1, 0u);
      }
    }
  } else if (a.mode == 12) {
    if (tid == 0) {
      for (int i = 0; i < a.iters; ++i) {
        fence_sc_gpu();
      }
    }
  } else if (a.mode == 13) {
    if (tid == 0) {
      for (int i = 0; i < a.iters; ++i) {
        acc += (float)ld_acquire_gpu_u32(a.sync);
      }
    }
  } else if (a.mode == 14) {
    if (tid == 0) {
      for (int i = 0; i < a.iters; ++i) {
        asm volatile("fence.sc.sys;" : : : "memory");
      }
    }
  } else if (a.mode == 15) {
    // System-scope relaxed store + acquire load of one word (doorbell / flag).
    if (tid == 0) {
      uint32_t* word = reinterpret_cast<uint32_t*>(const_cast<float*>(a.data));
      for (int i = 0; i < a.iters; ++i) {
        uint32_t seen;
        asm volatile("st.relaxed.sys.global.u32 [%0], %1;" : : "l"(word), "r"(i)
                     : "memory");
        asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
                     : "=r"(seen)
                     : "l"(word)
                     : "memory");
        acc += (float)seen;
      }
    }
  } else if (a.mode == 16) {
    // Plain volatile-free relaxed GPU-scope load spin (cheapest polling).
    if (tid == 0) {
      for (int i = 0; i < a.iters; ++i) {
        acc += (float)ld_relaxed_gpu_u32(a.sync);
      }
    }
  }
  else if (a.mode == 17) {
    // Plain 16-byte stores, block-major contiguous packs: pack g of the launch
    // goes to thread g (what a collective's staging loop does).
    const int64_t g = (int64_t)blockIdx.x * blockDim.x + tid;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    float4* p = reinterpret_cast<float4*>(const_cast<float*>(a.data));
    for (int64_t i = g; i < (int64_t)a.iters; i += stride) {
      p[i] = make_float4(acc, acc, acc, acc);
    }
  } else if (a.mode == 18) {
    // Same pattern, system-scope loads (a collective's reduce / gather loop).
    const int64_t g = (int64_t)blockIdx.x * blockDim.x + tid;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    const uint32_t* p = reinterpret_cast<const uint32_t*>(a.data);
    for (int64_t i = g; i < (int64_t)a.iters; i += stride) {
      uint32_t w0, w1, w2, w3;
      asm volatile("ld.relaxed.sys.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                   : "=r"(w0), "=r"(w1), "=r"(w2), "=r"(w3)
                   : "l"(p + 4 * i)
                   : "memory");
      acc += (float)(w0 ^ w1 ^ w2 ^ w3);
    }
  } else if (a.mode == 19) {
    // Same pattern, plain loads.
    const int64_t g = (int64_t)blockIdx.x * blockDim.x + tid;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    const float4* p = reinterpret_cast<const float4*>(a.data);
    for (int64_t i = g; i < (int64_t)a.iters; i += stride) {
      const float4 v = p[i];
      acc += v.x + v.y + v.z + v.w;
    }
  }
  if (tid == 0) {
    a.out[blockIdx.x] = acc;
  }
}

// Thread-block cluster probe: cluster-wide barrier plus a distributed
// shared-memory read of the cluster's first block.
__global__ void __cluster_dims__(8, 1, 1) probe_cluster_kernel(
    const ProbeArgs a) {
  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();
  __shared__ float s_val[32];
  const int tid = threadIdx.x;
  float acc = (float)tid;
  for (int i = 0; i < a.iters; ++i) {
    if (tid == 0) {
      s_val[0] = acc + (float)cluster.block_rank();
    }
    cluster.sync();
    if (a.mode == 1) {
      const float* remote = cluster.map_shared_rank(s_val, 0);
      acc += remote[0];
      cluster.sync();
    }
  }
  if (tid == 0) {
    a.out[blockIdx.x] = acc;
  }
}

inline void launch_probe_cluster(const ProbeArgs& args, int grid, int threads,
                                 cudaStream_t stream) {
  probe_cluster_kernel<<<dim3(grid, 1, 1), dim3(threads, 1, 1), 0, stream>>>(
      args);
}

inline void launch_probe(const ProbeArgs& args, int grid, int threads,
                         cudaStream_t stream) {
  probe_kernel<<<dim3(grid, 1, 1), dim3(threads, 1, 1), 0, stream>>>(args);
}

}  // namespace k31sm121
