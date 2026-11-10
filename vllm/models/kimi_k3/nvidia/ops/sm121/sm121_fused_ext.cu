// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// JIT-built bindings for the Kimi-K3 SM12x fused kernels (see _ext.py).
// The Python wrappers validate dtypes, shapes, strides and alignment; this
// file only unpacks pointers and launches on the current CUDA stream.

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "attn_res.cuh"
#include "kda_decode.cuh"
#include "probe.cuh"
#include "roce_fused.cuh"
#include "roce_gemm.cuh"

namespace {

template <typename T>
T* ptr(const at::Tensor& tensor) {
  return reinterpret_cast<T*>(tensor.data_ptr());
}

template <typename T>
T* optional_ptr(const c10::optional<at::Tensor>& tensor) {
  return tensor.has_value() ? reinterpret_cast<T*>(tensor->data_ptr()) : nullptr;
}

cudaStream_t current_stream(const at::Tensor& tensor) {
  return c10::cuda::getCurrentCUDAStream(tensor.get_device()).stream();
}

void kda_fused_decode(
    const at::Tensor& x, int64_t x_row_stride, const at::Tensor& conv_w,
    const at::Tensor& conv_state, int64_t cs_slot_stride,
    int64_t cs_chan_stride, int64_t cs_tap_stride, const at::Tensor& raw_g,
    int64_t g_row_stride, const at::Tensor& raw_beta, int64_t beta_row_stride,
    const at::Tensor& a_log, const at::Tensor& dt_bias, const at::Tensor& state,
    int64_t state_slot_stride, const at::Tensor& state_indices,
    int64_t state_indices_stride,
    const c10::optional<at::Tensor>& num_accepted,
    const c10::optional<at::Tensor>& cu_seqlens, const at::Tensor& out_gate,
    int64_t og_row_stride, const at::Tensor& norm_w, const at::Tensor& out,
    int64_t out_row_stride, int64_t num_seqs, int64_t num_heads,
    double lower_bound, double scale, double norm_eps) {
  const c10::cuda::CUDAGuard guard(x.device());
  k31sm121::KdaDecodeArgs args;
  args.x = ptr<const __nv_bfloat16>(x);
  args.x_row_stride = x_row_stride;
  args.conv_w = ptr<const float>(conv_w);
  args.conv_state = ptr<__nv_bfloat16>(conv_state);
  args.cs_slot_stride = cs_slot_stride;
  args.cs_chan_stride = cs_chan_stride;
  args.cs_tap_stride = cs_tap_stride;
  args.raw_g = ptr<const __nv_bfloat16>(raw_g);
  args.g_row_stride = g_row_stride;
  args.raw_beta = ptr<const __nv_bfloat16>(raw_beta);
  args.beta_row_stride = beta_row_stride;
  args.a_log = ptr<const float>(a_log);
  args.dt_bias = ptr<const float>(dt_bias);
  args.state = ptr<float>(state);
  args.state_slot_stride = state_slot_stride;
  args.conv_indices = ptr<const int32_t>(state_indices);
  args.conv_indices_stride = state_indices_stride;
  args.state_indices = ptr<const int32_t>(state_indices);
  args.state_indices_stride = state_indices_stride;
  args.num_accepted = optional_ptr<const int32_t>(num_accepted);
  args.cu_seqlens = optional_ptr<const int32_t>(cu_seqlens);
  args.out_gate = ptr<const __nv_bfloat16>(out_gate);
  args.og_row_stride = og_row_stride;
  args.norm_w = ptr<const float>(norm_w);
  args.out = ptr<__nv_bfloat16>(out);
  args.out_row_stride = out_row_stride;
  args.H = static_cast<int>(num_heads);
  args.LP = static_cast<int>(num_heads) * k31sm121::kKdaDim;
  args.lower_bound = static_cast<float>(lower_bound);
  args.scale = static_cast<float>(scale);
  args.norm_eps = static_cast<float>(norm_eps);
  k31sm121::launch_kda_fused_decode(args, static_cast<int>(num_seqs),
                                    current_stream(x));
}

void attn_res(
    const at::Tensor& prefix, int64_t prefix_row_stride,
    const c10::optional<at::Tensor>& delta, int64_t delta_row_stride,
    const at::Tensor& blocks, int64_t block_row_stride,
    int64_t block_slot_stride, const at::Tensor& norm_weight,
    const at::Tensor& qk_weight,
    const c10::optional<at::Tensor>& output_norm_weight,
    const at::Tensor& output, int64_t output_row_stride,
    const at::Tensor& partial, int64_t rows, int64_t num_blocks,
    int64_t block_write_idx, double eps, double output_norm_eps) {
  const c10::cuda::CUDAGuard guard(prefix.device());
  k31sm121::AttnResArgs args;
  args.prefix = ptr<__nv_bfloat16>(prefix);
  args.prefix_row_stride = prefix_row_stride;
  args.delta = optional_ptr<const __nv_bfloat16>(delta);
  args.delta_row_stride = delta_row_stride;
  args.blocks = ptr<__nv_bfloat16>(blocks);
  args.block_row_stride = block_row_stride;
  args.block_slot_stride = block_slot_stride;
  args.norm_weight = ptr<const __nv_bfloat16>(norm_weight);
  args.qk_weight = ptr<const __nv_bfloat16>(qk_weight);
  args.output_norm_weight = optional_ptr<const __nv_bfloat16>(output_norm_weight);
  args.output = ptr<__nv_bfloat16>(output);
  args.output_row_stride = output_row_stride;
  if (partial.numel() <
      rows * k31sm121::kAttnResBlocks * k31sm121::kAttnResPartials) {
    throw std::runtime_error("attn_res: scratch is smaller than the batch");
  }
  args.partial = ptr<float>(partial);
  args.num_blocks = static_cast<int>(num_blocks);
  args.block_write_idx = static_cast<int>(block_write_idx);
  args.eps = static_cast<float>(eps);
  args.output_norm_eps = static_cast<float>(output_norm_eps);
  k31sm121::launch_attn_res(args, static_cast<int>(rows),
                            current_stream(prefix));
}

void probe(const at::Tensor& data, const at::Tensor& out,
           const at::Tensor& sync_words, int64_t mode, int64_t iters,
           int64_t group, int64_t grid, int64_t threads) {
  // `data` may be pinned host memory (device-addressable on an integrated GPU).
  const c10::cuda::CUDAGuard guard(out.device());
  k31sm121::ProbeArgs args;
  args.data = ptr<const float>(data);
  args.out = ptr<float>(out);
  args.sync = ptr<uint32_t>(sync_words);
  args.mode = static_cast<int>(mode);
  args.iters = static_cast<int>(iters);
  args.group = static_cast<int>(group);
  if (mode >= 100) {
    args.mode = static_cast<int>(mode - 100);
    k31sm121::launch_probe_cluster(args, static_cast<int>(grid),
                                   static_cast<int>(threads),
                                   current_stream(out));
    return;
  }
  k31sm121::launch_probe(args, static_cast<int>(grid),
                         static_cast<int>(threads), current_stream(out));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("kda_fused_decode", &kda_fused_decode);
  m.def("attn_res", &attn_res);
  m.def("probe", &probe);
  k31sm121::bind_roce_fused(m);
  k31sm121::bind_roce_gemm(m);
}
