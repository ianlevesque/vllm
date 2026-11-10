// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// RoCEnante fused collectives for Kimi-K3 on SM12x (multi-node DGX Spark TP).
//
// The SM100 decode path fuses its tensor-parallel collectives with the math
// around them (GEMM + all-reduce, and the latent-MoE tail: latent all-reduce +
// RMSNorm + shared reduce-scatter, then a column-sharded up-projection whose
// result is multicast).  Those kernels write peers' memory over NVLink.  A
// Spark rank has no peer mapping: its collectives are RDMA writes posted by the
// b12x RoCEnante proxy, and their latency grows with the bytes each rank sends
// to each peer (measured: ~20 us per round plus ~70 ns per KB and peer).  The
// kernels here are the same fusions rebuilt on that transport, using the
// proxy's windowed ops so that a rank sends a peer only the columns that peer
// owns:
//
//   two_shot_all_reduce   reduce-scatter + all-gather of [rows, N] BF16 in one
//                         launch: 2 rounds of N/world columns per peer instead
//                         of one round of N (or the FP32 4x4 hierarchy).
//   moe_tail_collective   the latent-MoE tail's collective in one launch:
//                         round 1 reduce-scatters the routed latent [rows, D]
//                         and the shared-expert partial [rows, N] (each rank
//                         keeps the sum of its own D/world and N/world
//                         columns); round 2 all-gathers the reduced latent and
//                         applies the RMSNorm to it.
//   all_gather2           last-dim all-gather of one or two [rows, w] tensors
//                         in one round (the MoE front's router logits and
//                         latent down-projection shard travel together).
//
// Protocol (b12x.comm.roce, unchanged): read the device epoch, stage into
// send[seq & 1], publish the slot's doorbell word and then seq, wait for
// flag[peer][seq & 1][hca] == seq of every peer and HCA, consume
// recv[peer][seq & 1], advance the epoch last.  A launch here may consume two
// consecutive sequences.
//
// Each kernel is one 8-block thread-block cluster.  On GB10 a cluster barrier
// costs ~0.23 us against ~1.9 us for a barrier built from GPU-scope atomics and
// fences, an SM stores only ~7 GB/s, and system-scope loads of NIC-written
// memory are latency-bound per warp, so the work is spread over 8 SMs and the
// blocks meet at cluster barriers; only block 0 rings the doorbell, polls the
// flags and advances the epoch.

#pragma once

#include <cooperative_groups.h>
#include <torch/extension.h>
#include <vector>

#include "sm121_common.cuh"

namespace k31sm121 {

constexpr int kRoceBlocks = 8;          // one cluster
constexpr int kRoceThreads = 512;
constexpr int kRoceWarps = kRoceThreads / 32;
constexpr int kRoceStride = kRoceBlocks * kRoceThreads;   // 4096 threads
constexpr int kRoceMaxWorld = 16;
constexpr uint32_t kRoceWindowed = 0x80000000u;
// Windowed-format op posted on the first HCA only (small payloads: half the
// RDMA writes and flags of a round).  The kernel then waits on that rail only.
constexpr uint32_t kRoceSingleRail = 0x40000000u;
constexpr uint32_t kRoceMaxWindowPacks = 0x3FFFu;
constexpr uint32_t kRoceMaxCommonPacks = 0xFFFFu;
// Per-peer payload up to which one rail is used.  Measured on 16 Sparks: one
// rail saves 1-5 us per round up to ~2.7 KB per peer and costs 2-4 us at
// 5.4 KB, where the two NICs' parallel writes matter more than the 15 posts.
constexpr uint32_t kRoceSingleRailMaxPacks = 192;   // 3 KB
constexpr int kRoceTailMaxRows = 9;     // rows * 14 warps <= 128 warps (D = 3584)
constexpr int kRoceTailScratch = kRoceBlocks * kRoceWarps;  // floats

struct RoceHandles {
  uint64_t recv_base;    // pinned host memory, device-addressable
  uint64_t flag_base;
  uint64_t send_base;
  uint64_t ctrl_base;
  uint64_t slot_bytes;
  uint32_t* epoch;       // device words owned by the runtime
  uint32_t* poison;
  uint32_t spin_limit;
  int world;
  int rank;
  int hca_count;
  int slots;
  int flag_stride;
};

// ---- system-scope accesses to the pinned region ----------------------------

__device__ __forceinline__ void fence_sc_sys() {
  asm volatile("fence.sc.sys;" : : : "memory");
}

__device__ __forceinline__ void st_relaxed_sys_u32(uint64_t addr, uint32_t v) {
  asm volatile("st.relaxed.sys.global.u32 [%0], %1;" : : "l"(addr), "r"(v)
               : "memory");
}

__device__ __forceinline__ uint32_t ld_relaxed_sys_u32(uint64_t addr) {
  uint32_t v;
  asm volatile("ld.relaxed.sys.global.u32 %0, [%1];"
               : "=r"(v)
               : "l"(addr)
               : "memory");
  return v;
}

__device__ __forceinline__ uint32_t ld_acquire_sys_u32(uint64_t addr) {
  uint32_t v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
               : "=r"(v)
               : "l"(addr)
               : "memory");
  return v;
}

// One 16-byte pack of NIC-written memory, never served from a stale cache.
__device__ __forceinline__ Bf16Pack ld_sys_pack(uint64_t addr) {
  Bf16Pack p;
  asm volatile("ld.relaxed.sys.global.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(p.w[0]), "=r"(p.w[1]), "=r"(p.w[2]), "=r"(p.w[3])
               : "l"(addr)
               : "memory");
  return p;
}

__device__ __forceinline__ void st_pack(uint64_t addr, const Bf16Pack& p) {
  asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};"
               :
               : "l"(addr), "r"(p.w[0]), "r"(p.w[1]), "r"(p.w[2]), "r"(p.w[3])
               : "memory");
}

__device__ __forceinline__ uint64_t roce_send(const RoceHandles& h,
                                              uint32_t seq) {
  return h.send_base + (uint64_t)(seq & 1u) * h.slot_bytes;
}

__device__ __forceinline__ uint64_t roce_recv(const RoceHandles& h, int src,
                                              uint32_t seq) {
  return h.recv_base +
         ((uint64_t)src * h.slots + (seq & 1u)) * h.slot_bytes;
}

// Doorbell word of an op that sends every peer `common_packs` packs followed
// by its own `window_packs` packs.  Always the windowed format (a plain byte
// count cannot ask for one rail); small payloads use one rail.
__device__ __forceinline__ uint32_t roce_word(uint32_t common_packs,
                                              uint32_t window_packs) {
  uint32_t word = kRoceWindowed | (window_packs << 16) | common_packs;
  if (common_packs + window_packs <= kRoceSingleRailMaxPacks) {
    word |= kRoceSingleRail;
  }
  return word;
}

__device__ __forceinline__ int roce_rails(const RoceHandles& h, uint32_t word) {
  return (word & kRoceSingleRail) != 0u ? 1 : h.hca_count;
}

// A round is published in two steps so that one system-scope fence per block
// covers both the staged bytes and the size word: the lead thread stores the
// slot's doorbell word while the blocks stage, every block's thread 0 fences
// after its own stores, the cluster meets, and the lead thread rings.
__device__ __forceinline__ void roce_publish_word(const RoceHandles& h,
                                                  uint32_t seq, uint32_t word) {
  st_relaxed_sys_u32(h.ctrl_base + 4, word);
  st_relaxed_sys_u32(h.ctrl_base + 16 + (uint64_t)(seq & 1u) * 4, word);
}

__device__ __forceinline__ void roce_ring(const RoceHandles& h, uint32_t seq) {
  st_relaxed_sys_u32(h.ctrl_base, seq);
}

// Thread t < world * rails of the polling block waits for one stripe flag.  A
// wait that exceeds the spin limit records the failure exactly as the b12x
// kernels do and poisons the runtime; callers re-read the poison word after
// the next cluster barrier.
__device__ __forceinline__ void roce_wait_flag(const RoceHandles& h,
                                               uint32_t seq, int rails) {
  const int tid = threadIdx.x;
  if (tid >= h.world * rails) {
    return;
  }
  const int peer = tid / rails;
  const int hca = tid - peer * rails;
  if (peer == h.rank) {
    return;
  }
  const uint64_t flag =
      h.flag_base +
      (((uint64_t)peer * h.slots + (seq & 1u)) * h.hca_count + hca) *
          (uint64_t)h.flag_stride;
  uint32_t polls = 0;
  while (ld_acquire_sys_u32(flag) != seq) {
    if (++polls >= h.spin_limit) {
      st_relaxed_sys_u32(h.ctrl_base + 12, (uint32_t)peer);
      st_relaxed_sys_u32(h.ctrl_base + 24, (uint32_t)hca);
      st_relaxed_sys_u32(h.ctrl_base + 8, seq);
      st_release_gpu_u32(h.poison, seq);
      return;
    }
  }
}

// The block that leads the cluster advances the epoch once every block is done
// with the receive slots.  A failed sequence keeps the epoch, which turns every
// later launch into a no-op until the host raises.
__device__ __forceinline__ void roce_finish(const RoceHandles& h,
                                            uint32_t last_seq) {
  fence_sc_gpu();
  if (ld_relaxed_sys_u32(h.ctrl_base + 8) == 0u) {
    st_release_gpu_u32(h.epoch, last_seq);
  }
}

// Sum one pack over the ranks in rank order (bit-identical on every rank).
// `own` is this rank's contribution; peers come from their receive slots at
// byte offset `offset`.  Every peer's pack is requested before any is
// consumed: a system-scope load of memory the NIC has just written takes about
// 0.6 us on GB10, and a warp that waited for each in turn spent 9 us here.  The
// conditions are uniform over the warp (world and rank), so nothing diverges.
__device__ __forceinline__ void roce_reduce_pack(const RoceHandles& h,
                                                 uint32_t seq,
                                                 const Bf16Pack& own,
                                                 uint64_t offset, float* acc) {
  Bf16Pack packs[kRoceMaxWorld];
#pragma unroll
  for (int src = 0; src < kRoceMaxWorld; ++src) {
    packs[src] = own;
    if (src < h.world && src != h.rank) {
      packs[src] = ld_sys_pack(roce_recv(h, src, seq) + offset);
    }
  }
#pragma unroll
  for (int src = 0; src < kRoceMaxWorld; ++src) {
    if (src < h.world) {
      float v[8];
      unpack_bf16(packs[src], v);
      if (src == 0) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          acc[i] = v[i];
        }
      } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          acc[i] += v[i];
        }
      }
    }
  }
}

// ---- two-shot all-reduce -----------------------------------------------------

struct RoceTwoShotArgs {
  RoceHandles h;
  const __nv_bfloat16* in;   // [rows, N], 16-byte aligned, contiguous
  __nv_bfloat16* out;        // [rows, N]
  int rows;
  int row_packs;             // N / 8
  int owner_packs;           // N / world / 8
};

__global__ __cluster_dims__(kRoceBlocks, 1, 1) __launch_bounds__(kRoceThreads)
void roce_two_shot_all_reduce_kernel(const RoceTwoShotArgs a) {
  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();
  const RoceHandles& h = a.h;
  const int tid = threadIdx.x;
  const bool lead = blockIdx.x == 0;
  // Pack i is handled by thread i mod 4096 in block-major order: a warp's
  // loads and stores then cover 512 contiguous bytes.  (Interleaving the
  // blocks instead, so that small payloads use all eight SMs, measured about
  // twice as slow: every warp access touches 32 scattered partial lines.)
  const int g = blockIdx.x * kRoceThreads + tid;

  if (ld_relaxed_gpu_u32(h.poison) != 0u) {
    return;
  }
  const uint32_t seq1 = ld_relaxed_gpu_u32(h.epoch) + 1u;
  const uint32_t seq2 = seq1 + 1u;
  const int total = a.rows * a.row_packs;
  const int window_packs = a.rows * a.owner_packs;
  const uint64_t window_bytes = (uint64_t)window_packs * 16;
  const uint64_t send1 = roce_send(h, seq1);
  const uint64_t send2 = roce_send(h, seq2);

  // Round 1 (reduce-scatter): peer p receives this rank's partial of the
  // columns p owns, row-major, as window p of a windowed op.
  for (int i = g; i < total; i += kRoceStride) {
    const int m = i / a.row_packs;
    const int c = i - m * a.row_packs;
    const int p = c / a.owner_packs;
    if (p != h.rank) {
      const int w = c - p * a.owner_packs;
      st_pack(send1 + (uint64_t)p * window_bytes +
                  (uint64_t)(m * a.owner_packs + w) * 16,
              load_pack(a.in + (int64_t)i * 8));
    }
  }
  const uint32_t word1 = roce_word(0u, (uint32_t)window_packs);
  if (lead && tid == 0) {
    roce_publish_word(h, seq1, word1);
  }
  __syncthreads();
  if (tid == 0) {
    fence_sc_sys();
  }
  cluster.sync();
  if (lead) {
    if (tid == 0) {
      roce_ring(h, seq1);
    }
    roce_wait_flag(h, seq1, roce_rails(h, word1));
  }
  cluster.sync();
  if (ld_relaxed_gpu_u32(h.poison) != 0u) {
    return;
  }

  // Reduce the columns this rank owns, straight into the next send slot: it
  // is both the all-gather payload and this rank's shard of the result.
  for (int j = g; j < window_packs; j += kRoceStride) {
    const int m = j / a.owner_packs;
    const int w = j - m * a.owner_packs;
    const Bf16Pack own = load_pack(
        a.in + ((int64_t)m * a.row_packs + (int64_t)h.rank * a.owner_packs + w) * 8);
    float acc[8];
    roce_reduce_pack(h, seq1, own, (uint64_t)j * 16, acc);
    st_pack(send2 + (uint64_t)j * 16, pack_bf16(acc));
  }
  const uint32_t word2 = roce_word((uint32_t)window_packs, 0u);
  if (lead && tid == 0) {
    roce_publish_word(h, seq2, word2);
  }
  __syncthreads();
  if (tid == 0) {
    fence_sc_sys();
  }
  cluster.sync();

  // Round 2 (all-gather) of the reduced shards.
  if (lead) {
    if (tid == 0) {
      roce_ring(h, seq2);
    }
    roce_wait_flag(h, seq2, roce_rails(h, word2));
  }
  cluster.sync();
  if (ld_relaxed_gpu_u32(h.poison) != 0u) {
    return;
  }
  for (int i = g; i < total; i += kRoceStride) {
    const int m = i / a.row_packs;
    const int c = i - m * a.row_packs;
    const int p = c / a.owner_packs;
    const int w = c - p * a.owner_packs;
    const uint64_t offset = (uint64_t)(m * a.owner_packs + w) * 16;
    const Bf16Pack pack =
        p == h.rank ? load_pack(reinterpret_cast<const __nv_bfloat16*>(send2 + offset))
                    : ld_sys_pack(roce_recv(h, p, seq2) + offset);
    store_pack(a.out + (int64_t)i * 8, pack);
  }
  cluster.sync();
  if (lead && tid == 0) {
    roce_finish(h, seq2);
  }
}

// ---- latent-MoE tail collective ---------------------------------------------

struct RoceMoeTailArgs {
  RoceHandles h;
  const __nv_bfloat16* latent;       // [rows, D] this rank's routed latent partial
  const __nv_bfloat16* shared;       // [rows, N] this rank's shared-expert partial
  const __nv_bfloat16* norm_weight;  // [D] or nullptr (sum only)
  __nv_bfloat16* latent_out;         // [rows, D] RMSNorm(sum over ranks)
  __nv_bfloat16* shared_out;         // [rows, N / world] sum of this rank's columns
  float* scratch;                    // [128] per-warp sums of squares
  int rows;
  int latent_packs;                  // D / 8, a multiple of 32
  int latent_owner_packs;            // D / world / 8
  int row_packs;                     // N / 8
  int owner_packs;                   // N / world / 8
  float eps;
};

__global__ __cluster_dims__(kRoceBlocks, 1, 1) __launch_bounds__(kRoceThreads)
void roce_moe_tail_collective_kernel(const RoceMoeTailArgs a) {
  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();
  const RoceHandles& h = a.h;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const bool lead = blockIdx.x == 0;
  // Pack i is handled by thread i mod 4096 in block-major order: a warp's
  // loads and stores then cover 512 contiguous bytes.  (Interleaving the
  // blocks instead, so that small payloads use all eight SMs, measured about
  // twice as slow: every warp access touches 32 scattered partial lines.)
  const int g = blockIdx.x * kRoceThreads + tid;
  const int gw = (blockIdx.x * kRoceThreads + tid) >> 5;   // warp within the cluster

  if (ld_relaxed_gpu_u32(h.poison) != 0u) {
    return;
  }
  const uint32_t seq1 = ld_relaxed_gpu_u32(h.epoch) + 1u;
  const uint32_t seq2 = seq1 + 1u;
  const int latent_total = a.rows * a.latent_packs;
  const int shared_total = a.rows * a.row_packs;
  const int latent_window = a.rows * a.latent_owner_packs;
  const int shared_window = a.rows * a.owner_packs;
  const int window_packs = latent_window + shared_window;
  const uint64_t window_bytes = (uint64_t)window_packs * 16;
  const uint64_t send1 = roce_send(h, seq1);
  const uint64_t send2 = roce_send(h, seq2);

  // Round 1 (reduce-scatter): window p holds this rank's partial of the latent
  // columns and of the shared columns that peer p owns.
  for (int i = g; i < latent_total; i += kRoceStride) {
    const int m = i / a.latent_packs;
    const int c = i - m * a.latent_packs;
    const int p = c / a.latent_owner_packs;
    if (p != h.rank) {
      const int w = c - p * a.latent_owner_packs;
      st_pack(send1 + (uint64_t)p * window_bytes +
                  (uint64_t)(m * a.latent_owner_packs + w) * 16,
              load_pack(a.latent + (int64_t)i * 8));
    }
  }
  for (int i = g; i < shared_total; i += kRoceStride) {
    const int m = i / a.row_packs;
    const int c = i - m * a.row_packs;
    const int p = c / a.owner_packs;
    if (p != h.rank) {
      const int w = c - p * a.owner_packs;
      st_pack(send1 + (uint64_t)p * window_bytes +
                  (uint64_t)(latent_window + m * a.owner_packs + w) * 16,
              load_pack(a.shared + (int64_t)i * 8));
    }
  }
  const uint32_t word1 = roce_word(0u, (uint32_t)window_packs);
  if (lead && tid == 0) {
    roce_publish_word(h, seq1, word1);
  }
  __syncthreads();
  if (tid == 0) {
    fence_sc_sys();
  }
  cluster.sync();
  if (lead) {
    if (tid == 0) {
      roce_ring(h, seq1);
    }
    roce_wait_flag(h, seq1, roce_rails(h, word1));
  }
  cluster.sync();
  if (ld_relaxed_gpu_u32(h.poison) != 0u) {
    return;
  }

  // Reduce this rank's columns.  The latent shard goes straight into the next
  // send slot (round 2's payload and this rank's part of the result); the
  // shared shard is this rank's output.  One code path with selected operands:
  // a warp that straddles the two parts must not run the peer loads twice.
  for (int j = g; j < window_packs; j += kRoceStride) {
    const bool is_latent = j < latent_window;
    const int k = is_latent ? j : j - latent_window;
    const int per_row = is_latent ? a.latent_owner_packs : a.owner_packs;
    const int row_packs = is_latent ? a.latent_packs : a.row_packs;
    const __nv_bfloat16* source = is_latent ? a.latent : a.shared;
    const int m = k / per_row;
    const int w = k - m * per_row;
    const Bf16Pack own = load_pack(
        source + ((int64_t)m * row_packs + (int64_t)h.rank * per_row + w) * 8);
    float acc[8];
    roce_reduce_pack(h, seq1, own, (uint64_t)j * 16, acc);
    const uint64_t dst =
        is_latent ? send2 + (uint64_t)j * 16
                  : reinterpret_cast<uint64_t>(a.shared_out) + (uint64_t)k * 16;
    st_pack(dst, pack_bf16(acc));
  }
  const uint32_t word2 = roce_word((uint32_t)latent_window, 0u);
  if (lead && tid == 0) {
    roce_publish_word(h, seq2, word2);
  }
  __syncthreads();
  if (tid == 0) {
    fence_sc_sys();
  }
  cluster.sync();

  // Round 2 (all-gather) of the reduced latent shards.
  if (lead) {
    if (tid == 0) {
      roce_ring(h, seq2);
    }
    roce_wait_flag(h, seq2, roce_rails(h, word2));
  }
  cluster.sync();
  if (ld_relaxed_gpu_u32(h.poison) != 0u) {
    return;
  }

  // Assemble the reduced latent and normalise it.  Warp gw owns 32 consecutive
  // packs of one row (a row is latent_packs / 32 whole warps), so a row's sum
  // of squares is a warp sum, a store, one cluster barrier, and a fixed-order
  // sum of the row's warps.  The reduced latent is BF16 here, exactly what the
  // unfused path hands from its all-reduce to its RMSNorm.
  const int warps_per_row = a.latent_packs >> 5;
  const int row = gw / warps_per_row;
  const bool active = row < a.rows;
  const int col_pack = (gw - row * warps_per_row) * 32 + lane;
  const int64_t pack_index = (int64_t)row * a.latent_packs + col_pack;
  float reduced[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    reduced[i] = 0.0f;
  }
  if (active) {
    const int p = col_pack / a.latent_owner_packs;
    const int w = col_pack - p * a.latent_owner_packs;
    const uint64_t offset = (uint64_t)(row * a.latent_owner_packs + w) * 16;
    const Bf16Pack pack =
        p == h.rank
            ? load_pack(reinterpret_cast<const __nv_bfloat16*>(send2 + offset))
            : ld_sys_pack(roce_recv(h, p, seq2) + offset);
    if (a.norm_weight == nullptr) {
      store_pack(a.latent_out + pack_index * 8, pack);
    } else {
      unpack_bf16(pack, reduced);
    }
  }
  if (a.norm_weight == nullptr) {
    cluster.sync();
  } else {
    float ssq = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      ssq += reduced[i] * reduced[i];
    }
    ssq = warp_sum(ssq);
    if (lane == 0) {
      a.scratch[gw] = ssq;
    }
    cluster.sync();
    if (active) {
      const float part =
          lane < warps_per_row ? a.scratch[row * warps_per_row + lane] : 0.0f;
      const float total = warp_sum(part);
      const float rstd = rsqrtf(total / (float)(a.latent_packs * 8) + a.eps);
      float w[8];
      unpack_bf16(load_pack(a.norm_weight + (int64_t)col_pack * 8), w);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        reduced[i] = (reduced[i] * rstd) * w[i];
      }
      store_pack(a.latent_out + pack_index * 8, pack_bf16(reduced));
    }
    cluster.sync();
  }
  if (lead && tid == 0) {
    roce_finish(h, seq2);
  }
}

// ---- all-gather of one or two tensors ----------------------------------------

struct RoceGather2Args {
  RoceHandles h;
  const __nv_bfloat16* in0;   // [rows, packs0 * 16 bytes] (any 16-byte packed dtype)
  const __nv_bfloat16* in1;   // optional second tensor, or nullptr
  __nv_bfloat16* out0;        // [rows, world * packs0 packs]
  __nv_bfloat16* out1;
  int rows;
  int packs0;                 // packs per row of in0
  int packs1;                 // packs per row of in1 (0 when absent)
};

__global__ __cluster_dims__(kRoceBlocks, 1, 1) __launch_bounds__(kRoceThreads)
void roce_all_gather2_kernel(const RoceGather2Args a) {
  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();
  const RoceHandles& h = a.h;
  const int tid = threadIdx.x;
  const bool lead = blockIdx.x == 0;
  // Pack i is handled by thread i mod 4096 in block-major order: a warp's
  // loads and stores then cover 512 contiguous bytes.  (Interleaving the
  // blocks instead, so that small payloads use all eight SMs, measured about
  // twice as slow: every warp access touches 32 scattered partial lines.)
  const int g = blockIdx.x * kRoceThreads + tid;

  if (ld_relaxed_gpu_u32(h.poison) != 0u) {
    return;
  }
  const uint32_t seq = ld_relaxed_gpu_u32(h.epoch) + 1u;
  const int shard0 = a.rows * a.packs0;
  const int shard1 = a.rows * a.packs1;
  const uint64_t send = roce_send(h, seq);

  for (int i = g; i < shard0; i += kRoceStride) {
    st_pack(send + (uint64_t)i * 16, load_pack(a.in0 + (int64_t)i * 8));
  }
  for (int i = g; i < shard1; i += kRoceStride) {
    st_pack(send + (uint64_t)(shard0 + i) * 16, load_pack(a.in1 + (int64_t)i * 8));
  }
  const uint32_t word = roce_word((uint32_t)(shard0 + shard1), 0u);
  if (lead && tid == 0) {
    roce_publish_word(h, seq, word);
  }
  __syncthreads();
  if (tid == 0) {
    fence_sc_sys();
  }
  cluster.sync();
  if (lead) {
    if (tid == 0) {
      roce_ring(h, seq);
    }
    roce_wait_flag(h, seq, roce_rails(h, word));
  }
  cluster.sync();
  if (ld_relaxed_gpu_u32(h.poison) != 0u) {
    return;
  }

  const int total0 = shard0 * h.world;
  const int total1 = shard1 * h.world;
  for (int i = g; i < total0 + total1; i += kRoceStride) {
    const bool second = i >= total0;
    const int k = second ? i - total0 : i;
    const int packs = second ? a.packs1 : a.packs0;
    const int row_packs = packs * h.world;
    const int m = k / row_packs;
    const int c = k - m * row_packs;
    const int p = c / packs;
    const int w = c - p * packs;
    const int64_t src_pack = (int64_t)m * packs + w;
    const Bf16Pack pack =
        p == h.rank
            ? load_pack((second ? a.in1 : a.in0) + src_pack * 8)
            : ld_sys_pack(roce_recv(h, p, seq) +
                          (uint64_t)((second ? shard0 : 0) + src_pack) * 16);
    store_pack((second ? a.out1 : a.out0) + (int64_t)k * 8, pack);
  }
  cluster.sync();
  if (lead && tid == 0) {
    roce_finish(h, seq);
  }
}

// ---- bindings -----------------------------------------------------------------

inline RoceHandles roce_handles(const std::vector<int64_t>& v) {
  if (v.size() != 13) {
    throw std::runtime_error("RoCE handle list must have 13 entries");
  }
  RoceHandles h;
  h.recv_base = static_cast<uint64_t>(v[0]);
  h.flag_base = static_cast<uint64_t>(v[1]);
  h.send_base = static_cast<uint64_t>(v[2]);
  h.ctrl_base = static_cast<uint64_t>(v[3]);
  h.slot_bytes = static_cast<uint64_t>(v[4]);
  h.epoch = reinterpret_cast<uint32_t*>(v[5]);
  h.poison = reinterpret_cast<uint32_t*>(v[6]);
  h.spin_limit = static_cast<uint32_t>(v[7]);
  h.world = static_cast<int>(v[8]);
  h.rank = static_cast<int>(v[9]);
  h.hca_count = static_cast<int>(v[10]);
  h.slots = static_cast<int>(v[11]);
  h.flag_stride = static_cast<int>(v[12]);
  return h;
}

inline cudaStream_t roce_stream(const at::Tensor& tensor) {
  return c10::cuda::getCurrentCUDAStream(tensor.get_device()).stream();
}

inline void roce_two_shot_all_reduce(const std::vector<int64_t>& handles,
                                     const at::Tensor& in,
                                     const at::Tensor& out, int64_t rows,
                                     int64_t columns) {
  const c10::cuda::CUDAGuard guard(in.device());
  RoceTwoShotArgs args;
  args.h = roce_handles(handles);
  args.in = reinterpret_cast<const __nv_bfloat16*>(in.data_ptr());
  args.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  args.rows = static_cast<int>(rows);
  args.row_packs = static_cast<int>(columns / 8);
  args.owner_packs = static_cast<int>(columns / args.h.world / 8);
  roce_two_shot_all_reduce_kernel<<<dim3(kRoceBlocks, 1, 1),
                                    dim3(kRoceThreads, 1, 1), 0,
                                    roce_stream(in)>>>(args);
}

inline void roce_moe_tail_collective(
    const std::vector<int64_t>& handles, const at::Tensor& latent,
    const at::Tensor& shared, const c10::optional<at::Tensor>& norm_weight,
    const at::Tensor& latent_out, const at::Tensor& shared_out,
    const at::Tensor& scratch, int64_t rows, int64_t latent_dim,
    int64_t columns, double eps) {
  const c10::cuda::CUDAGuard guard(latent.device());
  RoceMoeTailArgs args;
  args.h = roce_handles(handles);
  args.latent = reinterpret_cast<const __nv_bfloat16*>(latent.data_ptr());
  args.shared = reinterpret_cast<const __nv_bfloat16*>(shared.data_ptr());
  args.norm_weight =
      norm_weight.has_value()
          ? reinterpret_cast<const __nv_bfloat16*>(norm_weight->data_ptr())
          : nullptr;
  args.latent_out = reinterpret_cast<__nv_bfloat16*>(latent_out.data_ptr());
  args.shared_out = reinterpret_cast<__nv_bfloat16*>(shared_out.data_ptr());
  args.scratch = reinterpret_cast<float*>(scratch.data_ptr());
  args.rows = static_cast<int>(rows);
  args.latent_packs = static_cast<int>(latent_dim / 8);
  args.latent_owner_packs = static_cast<int>(latent_dim / args.h.world / 8);
  args.row_packs = static_cast<int>(columns / 8);
  args.owner_packs = static_cast<int>(columns / args.h.world / 8);
  args.eps = static_cast<float>(eps);
  roce_moe_tail_collective_kernel<<<dim3(kRoceBlocks, 1, 1),
                                    dim3(kRoceThreads, 1, 1), 0,
                                    roce_stream(latent)>>>(args);
}

// Byte-generic: tensors are moved as 16-byte packs, whatever their dtype.
inline void roce_all_gather2(const std::vector<int64_t>& handles,
                             const at::Tensor& in0,
                             const c10::optional<at::Tensor>& in1,
                             const at::Tensor& out0,
                             const c10::optional<at::Tensor>& out1,
                             int64_t rows, int64_t packs0, int64_t packs1) {
  const c10::cuda::CUDAGuard guard(in0.device());
  RoceGather2Args args;
  args.h = roce_handles(handles);
  args.in0 = reinterpret_cast<const __nv_bfloat16*>(in0.data_ptr());
  args.out0 = reinterpret_cast<__nv_bfloat16*>(out0.data_ptr());
  args.in1 = in1.has_value()
                 ? reinterpret_cast<const __nv_bfloat16*>(in1->data_ptr())
                 : nullptr;
  args.out1 = out1.has_value()
                  ? reinterpret_cast<__nv_bfloat16*>(out1->data_ptr())
                  : nullptr;
  args.rows = static_cast<int>(rows);
  args.packs0 = static_cast<int>(packs0);
  args.packs1 = static_cast<int>(packs1);
  roce_all_gather2_kernel<<<dim3(kRoceBlocks, 1, 1), dim3(kRoceThreads, 1, 1),
                            0, roce_stream(in0)>>>(args);
}

inline void bind_roce_fused(pybind11::module_& m) {
  m.def("roce_two_shot_all_reduce", &roce_two_shot_all_reduce);
  m.def("roce_moe_tail_collective", &roce_moe_tail_collective);
  m.def("roce_all_gather2", &roce_all_gather2);
}

}  // namespace k31sm121
