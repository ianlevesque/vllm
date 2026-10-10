// Fused sigmoid + biased top-16 routing for Kimi-K3 (896 experts).
//
// Bit-exact replacement for the moeSigmoid + moeTopK pair in
// csrc/libtorch_stable/moe/topk_softmax_kernels.cu for the K3 decode shape:
//   - scores   = 1/(1+__expf(-logit)), NaN/Inf clamped to 0 (same intrinsic)
//   - selection key = score + e_score_correction_bias
//   - output weight = unbiased score of the selected expert
//   - 16 sequential argmax passes; ties resolved to the LOWER expert index
//     (cub::ArgMax semantics)
//   - renormalization sums the unbiased scores in selection order and applies
//     scale = routed_scaling_factor / (sum > 0 ? sum : 1)
//   - padded rows emit indices = -1 while retaining the computed weights
//
// The reference walks global memory 16 times with a 16-deep prior-winner scan
// per element; this kernel stages both arrays in shared memory once and marks
// winners with -inf, which cannot collide with real keys (score in [0,1],
// finite bias).

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <torch/extension.h>

namespace kimi_topk16 {

constexpr int kExperts = 896;
constexpr int kTopK = 16;
constexpr int kThreads = 256;
constexpr int kPerThread = (kExperts + kThreads - 1) / kThreads;  // 4
constexpr int kWarps = kThreads / 32;

// Strictly-greater wins; on equal keys the lower expert index wins.
__device__ __forceinline__ void arg_merge(
    float& best_val, int& best_idx, float val, int idx) {
  if (val > best_val || (val == best_val && idx < best_idx)) {
    best_val = val;
    best_idx = idx;
  }
}

__global__ __launch_bounds__(kThreads) void topk16_sigmoid_kernel(
    const float* __restrict__ logits,   // [rows, 896]
    const float* __restrict__ bias,     // [896]
    float* __restrict__ out_weights,    // [rows, 16]
    int* __restrict__ out_indices,      // [rows, 16]
    const bool* __restrict__ is_padding,  // [rows] or nullptr
    const bool renormalize,
    const float routed_scaling_factor) {
  __shared__ float score_sm[kExperts];  // unbiased sigmoid scores
  __shared__ float key_sm[kExperts];    // selection keys (score + bias)
  __shared__ float warp_val[kWarps];
  __shared__ int warp_idx[kWarps];
  __shared__ float sel_weight[kTopK];
  __shared__ int sel_index[kTopK];

  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const float* row_logits = logits + static_cast<long>(row) * kExperts;

  for (int e = tid; e < kExperts; e += kThreads) {
    const float val = row_logits[e];
    float s = 1.0f / (1.0f + __expf(-val));
    if (isnan(s) || isinf(s)) {
      s = 0.0f;
    }
    score_sm[e] = s;
    key_sm[e] = s + bias[e];
  }
  __syncthreads();

  for (int k_idx = 0; k_idx < kTopK; ++k_idx) {
    float best_val = -CUDART_INF_F;
    int best_idx = kExperts;
#pragma unroll
    for (int i = 0; i < kPerThread; ++i) {
      const int e = tid + i * kThreads;
      if (e < kExperts) {
        arg_merge(best_val, best_idx, key_sm[e], e);
      }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      const float other_val = __shfl_down_sync(0xffffffffu, best_val, offset);
      const int other_idx = __shfl_down_sync(0xffffffffu, best_idx, offset);
      arg_merge(best_val, best_idx, other_val, other_idx);
    }
    if (lane == 0) {
      warp_val[warp] = best_val;
      warp_idx[warp] = best_idx;
    }
    __syncthreads();
    if (tid == 0) {
      float v = warp_val[0];
      int x = warp_idx[0];
#pragma unroll
      for (int w = 1; w < kWarps; ++w) {
        arg_merge(v, x, warp_val[w], warp_idx[w]);
      }
      sel_index[k_idx] = x;
      sel_weight[k_idx] = score_sm[x];
      key_sm[x] = -CUDART_INF_F;
    }
    __syncthreads();
  }

  if (tid == 0) {
    const bool pad = is_padding != nullptr && is_padding[row];
    float selected_sum = 0.0f;
    if (renormalize) {
      for (int k_idx = 0; k_idx < kTopK; ++k_idx) {
        selected_sum += sel_weight[k_idx];
      }
    }
    float scale = routed_scaling_factor;
    if (renormalize) {
      const float denom = selected_sum > 0.0f ? selected_sum : 1.0f;
      scale /= denom;
    }
    float* out_w = out_weights + static_cast<long>(row) * kTopK;
    int* out_i = out_indices + static_cast<long>(row) * kTopK;
    for (int k_idx = 0; k_idx < kTopK; ++k_idx) {
      out_w[k_idx] = sel_weight[k_idx] * scale;
      out_i[k_idx] = pad ? -1 : sel_index[k_idx];
    }
  }
}

}  // namespace kimi_topk16

static void topk16_sigmoid(
    torch::Tensor logits,
    torch::Tensor bias,
    torch::Tensor out_weights,
    torch::Tensor out_indices,
    c10::optional<torch::Tensor> is_padding,
    bool renormalize,
    double routed_scaling_factor) {
  TORCH_CHECK(logits.is_cuda() && logits.dtype() == torch::kFloat32 &&
                  logits.dim() == 2 &&
                  logits.size(1) == kimi_topk16::kExperts &&
                  logits.is_contiguous(),
              "logits must be contiguous CUDA FP32 [rows, 896]");
  TORCH_CHECK(bias.is_cuda() && bias.dtype() == torch::kFloat32 &&
                  bias.numel() == kimi_topk16::kExperts &&
                  bias.is_contiguous(),
              "bias must be contiguous CUDA FP32 [896]");
  const long rows = logits.size(0);
  TORCH_CHECK(out_weights.is_cuda() && out_weights.dtype() == torch::kFloat32 &&
                  out_weights.is_contiguous() &&
                  out_weights.numel() == rows * kimi_topk16::kTopK,
              "out_weights must be contiguous CUDA FP32 [rows, 16]");
  TORCH_CHECK(out_indices.is_cuda() && out_indices.dtype() == torch::kInt32 &&
                  out_indices.is_contiguous() &&
                  out_indices.numel() == rows * kimi_topk16::kTopK,
              "out_indices must be contiguous CUDA INT32 [rows, 16]");
  const bool* pad_ptr = nullptr;
  if (is_padding.has_value()) {
    TORCH_CHECK(is_padding->is_cuda() && is_padding->dtype() == torch::kBool &&
                    is_padding->is_contiguous() && is_padding->numel() >= rows,
                "is_padding must be contiguous CUDA bool [rows]");
    pad_ptr = is_padding->data_ptr<bool>();
  }
  if (rows == 0) {
    return;
  }
  const c10::cuda::CUDAGuard guard(logits.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  kimi_topk16::topk16_sigmoid_kernel<<<
      static_cast<int>(rows), kimi_topk16::kThreads, 0, stream>>>(
      logits.data_ptr<float>(),
      bias.data_ptr<float>(),
      out_weights.data_ptr<float>(),
      out_indices.data_ptr<int>(),
      pad_ptr,
      renormalize,
      static_cast<float>(routed_scaling_factor));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("topk16_sigmoid", &topk16_sigmoid);
}
