// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Small-M BF16 GEMM fused with its tensor-parallel collective, for Kimi-K3 on
// SM12x over RoCEnante (see roce_fused.cuh for the transport).
//
// The SM100 path has two such fusions: GEMM-AR (a row-parallel projection
// whose partial product is reduced as it is produced) and the latent-MoE
// tail's fused-add up-projection (a column-sharded projection added to the
// shared-expert shard and multicast).  Both write the GEMM's result straight
// into transport memory.  Here that memory is the RoCEnante send slot:
//
//   gemm_two_shot_all_reduce   out[rows, N] = sum over ranks of
//                              x_r[rows, K] @ W_r[N, K]^T.  The partial is
//                              rounded to BF16 into the reduce-scatter windows
//                              of the send slot; the rounds are those of
//                              roce_two_shot_all_reduce.
//   gemm_add_all_gather        shard[rows, N/world] = addend + x @ W^T with
//                              W the rank's [N/world, K] column shard, written
//                              into the send slot as the payload of one
//                              all-gather round.
//
// Neither needs an intermediate tensor, a staging copy, or a second launch.
//
// The GEMM.  rows <= 16, so one m16n8k16 tensor-core MMA covers every row of
// x for eight output columns and sixteen input features.  A warp owns one
// group of eight output columns (one 16-byte pack of each output row) for one
// segment of K.  The MMA's operand layout puts weight row n = column group *
// 8 + (lane / 4) in the four lanes of a lane group, so at each step those four
// lanes load one 64-byte line of that weight row and the warp streams eight
// weight rows; x is held in shared memory as BF16, and lane group g supplies
// rows g and g + 8 of it.  Because a dot product does not care about the order
// of its terms, each lane feeds its own 16-byte pack to two consecutive MMAs
// (features are permuted identically in both operands), which keeps every
// load a whole pack.  Long projections are split into segments over the warps
// of a block and summed in a fixed order.  The accumulators of lane group g
// are output row g (and g + 8) of the warp's pack, so the result is rounded
// and stored as four 4-byte pairs per row with no shuffle.
//
// The grid is whole 8-block clusters.  It meets at a one-way barrier of
// GPU-scope atomics (once per launch); cluster 0 then runs the collective's
// rounds exactly as the unfused kernels do and the other clusters exit.
//
// Numerics: tensor-core FP32 accumulation of exact BF16 products, segments
// summed in order, an addend used as the initial accumulator, one rounding to
// BF16.  The all-reduce then sums the BF16 partials in rank order in FP32 and
// rounds once, like the unfused two-shot kernel.

#pragma once

#include "roce_fused.cuh"

namespace k31sm121 {

constexpr int kGemmMaxRows = 16;
constexpr int kGemmModeTwoShot = 0;
constexpr int kGemmModeGather = 1;
constexpr int kGemmStepElems = 32;        // features one warp step covers
// Measured on GB10 (16 Sparks, in CUDA graphs): an SM delivers ~27 GB/s of
// cached weights to its warps, so the GEMM wants many SMs, but every cluster
// beyond four costs more in launch and barrier than its SMs return (the
// 11 MB output projection at one row ran 6 us faster on 32 blocks than on 48
// and 8 us faster than on 56).
constexpr int kGemmMaxBlocks = 32;                       // short projections
constexpr int kGemmSegmentedTasks = 32 * kRoceWarps;     // long ones: 32 blocks
// Per-warp accumulators exchanged between the segments of a block.
constexpr size_t kGemmBounceBytes = kRoceThreads * 4 * sizeof(float);

struct RoceGemmArgs {
  RoceHandles h;
  const __nv_bfloat16* x;        // [rows, K], 16-byte aligned, contiguous
  const __nv_bfloat16* weight;   // [columns, K] row-major, 16-byte aligned
  const __nv_bfloat16* addend;   // [rows, columns] or nullptr
  __nv_bfloat16* out;            // two-shot: [rows, columns]; gather: [rows, world * columns]
  uint32_t* arrived;             // grid barrier word, zero between launches
  int rows;
  int k_packs;                   // K / 8
  int col_packs;                 // columns / 8: output packs per row = column groups
  int segments;                  // warps sharing one column group (a power of two <= 16)
  int gemm_blocks;               // blocks that share the column groups (segments == 1)
  int steps_per_segment;         // K / 32 / segments
  int blocks;                    // grid size, a multiple of the cluster size
  int mode;
};

// D = A (16 x 16, rows m) * B (16 x 8, columns n) + C, BF16 operands, FP32
// accumulators (PTX ISA "mma.sync.aligned.m16n8k16", as CUTLASS emits it).
__device__ __forceinline__ void gemm_mma(float* c, uint32_t a0, uint32_t a1,
                                         uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  float d0, d1, d2, d3;
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c[0]),
        "f"(c[1]), "f"(c[2]), "f"(c[3]));
  c[0] = d0;
  c[1] = d1;
  c[2] = d2;
  c[3] = d3;
}

// One step: this lane's weight pack and its packs of x rows g and g + 8 feed
// two MMAs (pairs 0, 1 and pairs 2, 3 of the packs).
__device__ __forceinline__ void gemm_step(float* c, const uint4& w,
                                          const uint4& lo, const uint4& hi) {
  gemm_mma(c, lo.x, hi.x, lo.y, hi.y, w.x, w.y);
  gemm_mma(c, lo.z, hi.z, lo.w, hi.w, w.z, w.w);
}

__device__ __forceinline__ uint32_t gemm_round_pair(float first, float second) {
  __nv_bfloat162 pair;
  pair.x = __float2bfloat16(first);
  pair.y = __float2bfloat16(second);
  return *reinterpret_cast<uint32_t*>(&pair);
}

// What a lane contributes to its warp's MMAs: weight row `group` of a column
// group and rows `group` and `group + 8` of x, at pack `slot` of every step.
// A lane group beyond the batch reads row 0 instead: its products only reach
// accumulator rows that are never stored.
struct GemmLane {
  int group;
  int slot;
  bool has_lo;
  bool has_hi;
  const uint4* x_lo;
  const uint4* x_hi;
};

__device__ __forceinline__ GemmLane gemm_lane(const RoceGemmArgs& a,
                                              const uint4* xs, int lane) {
  GemmLane ln;
  ln.group = lane >> 2;
  ln.slot = lane & 3;
  ln.has_lo = ln.group < a.rows;
  ln.has_hi = ln.group + 8 < a.rows;
  ln.x_lo = xs + (ln.has_lo ? ln.group : 0) * a.k_packs + ln.slot;
  ln.x_hi = xs + (ln.has_hi ? ln.group + 8 : 0) * a.k_packs + ln.slot;
  return ln;
}

// The addend is the initial accumulator: columns 2 * slot, + 1 of the pack.
__device__ __forceinline__ void gemm_init(const RoceGemmArgs& a,
                                          const GemmLane& ln, int col_pack,
                                          float* c) {
  c[0] = c[1] = c[2] = c[3] = 0.0f;
  if (a.addend == nullptr) {
    return;
  }
  const int64_t column = (int64_t)col_pack * 8 + ln.slot * 2;
  const int64_t row_elems = (int64_t)a.col_packs * 8;
  if (ln.has_lo) {
    const float2 v = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(
        a.addend + ln.group * row_elems + column));
    c[0] = v.x;
    c[1] = v.y;
  }
  if (ln.has_hi) {
    const float2 v = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(
        a.addend + (ln.group + 8) * row_elems + column));
    c[2] = v.x;
    c[3] = v.y;
  }
}

// Steps [first, last) of one column group.  Two weight loads are in flight
// per warp before either is consumed; rows 8-15 of x are loaded only when the
// batch has them (uniform over the warp).
__device__ __forceinline__ void gemm_run(const RoceGemmArgs& a,
                                         const GemmLane& ln, int col_pack,
                                         int first, int last, float* c) {
  const uint4* w = reinterpret_cast<const uint4*>(
                       a.weight +
                       ((int64_t)col_pack * 8 + ln.group) * (a.k_packs * 8)) +
                   ln.slot;
  const bool wide = a.rows > 8;
  const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
  int q = first;
  for (; q + 2 <= last; q += 2) {
    const uint4 w0 = w[4 * q];
    const uint4 w1 = w[4 * q + 4];
    const uint4 lo0 = ln.x_lo[4 * q];
    const uint4 lo1 = ln.x_lo[4 * q + 4];
    uint4 hi0 = zero;
    uint4 hi1 = zero;
    if (wide) {
      hi0 = ln.x_hi[4 * q];
      hi1 = ln.x_hi[4 * q + 4];
    }
    gemm_step(c, w0, lo0, hi0);
    gemm_step(c, w1, lo1, hi1);
  }
  if (q < last) {
    gemm_step(c, w[4 * q], ln.x_lo[4 * q], wide ? ln.x_hi[4 * q] : zero);
  }
}

// Round the lane's two values of rows `group` and `group + 8` and store them
// straight into the send slot (four lanes make one 16-byte pack of a row).
__device__ __forceinline__ void gemm_store(const RoceGemmArgs& a,
                                           const GemmLane& ln, int col_pack,
                                           uint64_t send, int owner_packs,
                                           uint64_t window_bytes,
                                           const float* c) {
  uint64_t pack_offset;   // of this pack within a row's region of the slot
  uint64_t row_stride;    // bytes between consecutive rows there
  if (a.mode == kGemmModeTwoShot) {
    // Window p of the reduce-scatter; the rank's own window is not sent and
    // holds its own partial for the reduction.
    const int p = col_pack / owner_packs;
    pack_offset = (uint64_t)p * window_bytes +
                  (uint64_t)(col_pack - p * owner_packs) * 16;
    row_stride = (uint64_t)owner_packs * 16;
  } else {
    pack_offset = (uint64_t)col_pack * 16;
    row_stride = (uint64_t)a.col_packs * 16;
  }
  const uint64_t dst = send + pack_offset + (uint64_t)ln.slot * 4;
  if (ln.has_lo) {
    asm volatile("st.global.u32 [%0], %1;"
                 :
                 : "l"(dst + (uint64_t)ln.group * row_stride),
                   "r"(gemm_round_pair(c[0], c[1]))
                 : "memory");
  }
  if (ln.has_hi) {
    asm volatile("st.global.u32 [%0], %1;"
                 :
                 : "l"(dst + (uint64_t)(ln.group + 8) * row_stride),
                   "r"(gemm_round_pair(c[2], c[3]))
                 : "memory");
  }
}

__global__ __cluster_dims__(kRoceBlocks, 1, 1) __launch_bounds__(kRoceThreads)
void roce_gemm_kernel(const RoceGemmArgs a) {
  namespace cg = cooperative_groups;
  extern __shared__ uint4 smem[];
  const RoceHandles& h = a.h;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const bool lead = blockIdx.x == 0;

  if (ld_relaxed_gpu_u32(h.poison) != 0u) {
    return;
  }
  const uint32_t seq1 = ld_relaxed_gpu_u32(h.epoch) + 1u;
  uint4* xs = smem;                                   // [rows][k_packs] BF16 packs
  float4* bounce = reinterpret_cast<float4*>(smem + a.rows * a.k_packs);

  for (int i = tid; i < a.rows * a.k_packs; i += kRoceThreads) {
    xs[i] = *reinterpret_cast<const uint4*>(a.x + (int64_t)i * 8);
  }
  __syncthreads();

  // The GEMM.  Short projections: the column groups are dealt evenly over
  // the blocks (an SM's load throughput is the bound) and over each block's
  // warps, and a warp stores a group as it finishes it.  Long projections:
  // `segments` warps share one column group and meet in shared memory.
  const uint64_t send1 = roce_send(h, seq1);
  // Two-shot geometry: every rank owns col_packs / world packs of a row.
  const int owner_packs = a.col_packs / h.world;
  const int window_packs = a.rows * owner_packs;
  const uint64_t window_bytes = (uint64_t)window_packs * 16;
  const GemmLane ln = gemm_lane(a, xs, lane);
  if (a.segments == 1) {
    int first_pack = 0;
    int end_pack = 0;
    if ((int)blockIdx.x < a.gemm_blocks) {
      first_pack = (int)((int64_t)blockIdx.x * a.col_packs / a.gemm_blocks);
      end_pack = (int)((int64_t)(blockIdx.x + 1) * a.col_packs / a.gemm_blocks);
    }
    // Uniform over the warp.
    for (int col_pack = first_pack + warp; col_pack < end_pack;
         col_pack += kRoceWarps) {
      float c[4];
      gemm_init(a, ln, col_pack, c);
      gemm_run(a, ln, col_pack, 0, a.steps_per_segment, c);
      gemm_store(a, ln, col_pack, send1, owner_packs, window_bytes, c);
    }
  } else {
    const int task = blockIdx.x * kRoceWarps + warp;
    const int segment = task & (a.segments - 1);
    const int col_pack = task / a.segments;
    const bool active = col_pack < a.col_packs;       // uniform over the warp
    float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (active) {
      if (segment == 0) {
        gemm_init(a, ln, col_pack, c);
      }
      const int first = segment * a.steps_per_segment;
      gemm_run(a, ln, col_pack, first, first + a.steps_per_segment, c);
    }
    bounce[tid] = make_float4(c[0], c[1], c[2], c[3]);
    __syncthreads();
    if (active && segment == 0) {
      // Sum the segments in order.
      float4 total = bounce[tid];
      for (int s = 1; s < a.segments; ++s) {
        const float4 part = bounce[tid + s * 32];
        total.x += part.x;
        total.y += part.y;
        total.z += part.z;
        total.w += part.w;
      }
      const float sum[4] = {total.x, total.y, total.z, total.w};
      gemm_store(a, ln, col_pack, send1, owner_packs, window_bytes, sum);
    }
  }
  const uint32_t word1 = a.mode == kGemmModeTwoShot
                             ? roce_word(0u, (uint32_t)window_packs)
                             : roce_word((uint32_t)(a.rows * a.col_packs), 0u);
  if (lead && tid == 0) {
    roce_publish_word(h, seq1, word1);
  }
  __syncthreads();
  // A grid of several clusters meets at the atomic barrier; one cluster needs
  // only its own cluster barrier below.
  const bool grid_barrier = a.blocks > kRoceBlocks;
  if (tid == 0) {
    fence_sc_sys();
    if (grid_barrier) {
      atom_add_gpu_u32(a.arrived, 1u);
    }
  }
  if (blockIdx.x >= kRoceBlocks) {
    return;
  }

  // Cluster 0 runs the collective once every block of the grid has stored.
  cg::cluster_group cluster = cg::this_cluster();
  const int g = blockIdx.x * kRoceThreads + tid;
  if (lead && tid == 0 && grid_barrier) {
    uint32_t polls = 0;
    while (ld_acquire_gpu_u32(a.arrived) != (uint32_t)a.blocks) {
      if (++polls >= h.spin_limit) {
        st_relaxed_sys_u32(h.ctrl_base + 12, (uint32_t)h.rank);
        st_relaxed_sys_u32(h.ctrl_base + 24, 0u);
        st_relaxed_sys_u32(h.ctrl_base + 8, seq1);
        st_release_gpu_u32(h.poison, seq1);
        break;
      }
    }
    st_release_gpu_u32(a.arrived, 0u);
  }
  cluster.sync();
  if (ld_relaxed_gpu_u32(h.poison) != 0u) {
    return;
  }
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

  if (a.mode == kGemmModeGather) {
    // The send slot holds this rank's shard; the peers' shards arrived.
    const int row_packs = a.col_packs * h.world;
    const int total = a.rows * row_packs;
    for (int i = g; i < total; i += kRoceStride) {
      const int m = i / row_packs;
      const int col = i - m * row_packs;
      const int p = col / a.col_packs;
      const int wcol = col - p * a.col_packs;
      const uint64_t offset = (uint64_t)(m * a.col_packs + wcol) * 16;
      const Bf16Pack pack =
          p == h.rank
              ? load_pack(reinterpret_cast<const __nv_bfloat16*>(send1 + offset))
              : ld_sys_pack(roce_recv(h, p, seq1) + offset);
      store_pack(a.out + (int64_t)i * 8, pack);
    }
    cluster.sync();
    if (lead && tid == 0) {
      roce_finish(h, seq1);
    }
    return;
  }

  // Two-shot all-reduce, from its reduction on (see roce_fused.cuh).
  const uint32_t seq2 = seq1 + 1u;
  const uint64_t send2 = roce_send(h, seq2);
  const __nv_bfloat16* own_window = reinterpret_cast<const __nv_bfloat16*>(
      send1 + (uint64_t)h.rank * window_bytes);
  for (int j = g; j < window_packs; j += kRoceStride) {
    float acc[8];
    roce_reduce_pack(h, seq1, load_pack(own_window + (int64_t)j * 8),
                     (uint64_t)j * 16, acc);
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
  const int total = a.rows * a.col_packs;
  for (int i = g; i < total; i += kRoceStride) {
    const int m = i / a.col_packs;
    const int col = i - m * a.col_packs;
    const int p = col / owner_packs;
    const int wcol = col - p * owner_packs;
    const uint64_t offset = (uint64_t)(m * owner_packs + wcol) * 16;
    const Bf16Pack pack =
        p == h.rank
            ? load_pack(reinterpret_cast<const __nv_bfloat16*>(send2 + offset))
            : ld_sys_pack(roce_recv(h, p, seq2) + offset);
    store_pack(a.out + (int64_t)i * 8, pack);
  }
  cluster.sync();
  if (lead && tid == 0) {
    roce_finish(h, seq2);
  }
}

// ---- launch --------------------------------------------------------------------

// The GEMM's shape on the grid.  A long, narrow projection is split into
// `segments` warps per column group (a power of two dividing both the block's
// 16 warps and the projection's steps); a short, wide one deals its column
// groups evenly over the blocks.
inline void gemm_shape(int64_t col_packs, int64_t steps, RoceGemmArgs* args) {
  int segments = 1;
  while (segments < kRoceWarps && steps % (segments * 2) == 0 &&
         col_packs * segments * 2 <= kGemmSegmentedTasks) {
    segments *= 2;
  }
  int64_t blocks = (col_packs * segments + kRoceWarps - 1) / kRoceWarps;
  if (segments == 1 && blocks > kGemmMaxBlocks) {
    blocks = kGemmMaxBlocks;
  }
  args->segments = segments;
  args->steps_per_segment = static_cast<int>(steps / segments);
  args->gemm_blocks = static_cast<int>(blocks);
  args->blocks =
      static_cast<int>((blocks + kRoceBlocks - 1) / kRoceBlocks * kRoceBlocks);
}

inline size_t gemm_shared_bytes(int64_t rows, int64_t k) {
  return static_cast<size_t>(rows * k) * sizeof(__nv_bfloat16) + kGemmBounceBytes;
}

// Dynamic shared memory the kernel may use.  Raising it above the default is
// an opt-in that roce_gemm_reserve makes once, before any CUDA graph capture.
inline size_t& gemm_reserved_bytes() {
  static size_t reserved = 48 * 1024;
  return reserved;
}

// Raises the kernel to the device's opt-in maximum (99 KB on GB10, 227 KB on
// data-centre parts) and returns it; the caller sizes x against it.
inline int64_t roce_gemm_reserve() {
  if (gemm_reserved_bytes() > 48 * 1024) {
    return static_cast<int64_t>(gemm_reserved_bytes());
  }
  int device = 0;
  int optin = 0;
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                             device) != cudaSuccess) {
    throw std::runtime_error("roce_gemm: cannot query the shared memory limit");
  }
  const size_t wanted = static_cast<size_t>(optin);
  if (wanted > gemm_reserved_bytes()) {
    const cudaError_t status = cudaFuncSetAttribute(
        roce_gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, optin);
    if (status != cudaSuccess) {
      throw std::runtime_error(
          std::string("roce_gemm: cannot reserve shared memory: ") +
          cudaGetErrorString(status));
    }
    gemm_reserved_bytes() = wanted;
  }
  return static_cast<int64_t>(gemm_reserved_bytes());
}

inline void roce_gemm(const std::vector<int64_t>& handles, const at::Tensor& x,
                      const at::Tensor& weight,
                      const c10::optional<at::Tensor>& addend,
                      const at::Tensor& out, const at::Tensor& arrived,
                      int64_t mode) {
  const c10::cuda::CUDAGuard guard(x.device());
  const int64_t rows = x.size(0);
  const int64_t k = x.size(1);
  const int64_t columns = weight.size(0);
  if (rows < 1 || rows > kGemmMaxRows || k < kGemmStepElems ||
      k % kGemmStepElems != 0 || weight.size(1) != k || columns % 8 != 0 ||
      (mode != kGemmModeTwoShot && mode != kGemmModeGather)) {
    throw std::runtime_error("roce_gemm: unsupported geometry");
  }
  RoceGemmArgs args;
  args.h = roce_handles(handles);
  args.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  args.weight = reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr());
  args.addend = addend.has_value()
                    ? reinterpret_cast<const __nv_bfloat16*>(addend->data_ptr())
                    : nullptr;
  args.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  args.arrived = reinterpret_cast<uint32_t*>(arrived.data_ptr());
  args.rows = static_cast<int>(rows);
  args.k_packs = static_cast<int>(k / 8);
  args.col_packs = static_cast<int>(columns / 8);
  gemm_shape(args.col_packs, k / kGemmStepElems, &args);
  args.mode = static_cast<int>(mode);
  const size_t shared_bytes = gemm_shared_bytes(rows, k);
  if (shared_bytes > gemm_reserved_bytes()) {
    throw std::runtime_error("roce_gemm: shared memory was not reserved");
  }
  roce_gemm_kernel<<<dim3(args.blocks, 1, 1), dim3(kRoceThreads, 1, 1),
                     shared_bytes, roce_stream(x)>>>(args);
}

inline void bind_roce_gemm(pybind11::module_& m) {
  m.def("roce_gemm", &roce_gemm);
  m.def("roce_gemm_reserve", &roce_gemm_reserve);
}

}  // namespace k31sm121
