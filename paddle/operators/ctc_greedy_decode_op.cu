/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "paddle/operators/ctc_greedy_decode_op.h"
#include "paddle/platform/cuda_helper.h"
#include "paddle/platform/gpu_info.h"

namespace paddle {
namespace operators {
using platform::PADDLE_CUDA_NUM_THREADS;

template <typename T>
__global__ void ArgmaxCudaKernel(const int seq_width, const T* logits,
                                 T* output) {
  T local_max_value = 0;
  int local_max_index = 0;
  __shared__ int max_value;

  for (int i = threadIdx.x; i < seq_width; i += BlockSize) {
    T value = logits[blockIdx.x * seq_width + i];
    if (value > local_max_value) {
      local_max_value = value;
      local_max_index = i;
    }
  }

  atomicMax(max_value, local_max_value);

  __syncthreads();

  if (local_max_value == max_value) {
    output[blockIdx.x] = local_max_index;
  }
}

template <typename T>
__global__ void MergeAndDelCudaKernel(const size_t num_token, const int* tokens,
                                      const size_t num_seq, const size_t* lod0,
                                      const int blank, bool merge_repeated,
                                      size_t* out_lod0, T* output) {
  int ouput_idx = 0;
  out_lod0[0] = 0;

  for (int i = 0; i < num_seq; ++i) {
    int pre_token = -1;
    for (int j = lod0[i]; j < lod0[i + 1]; ++j) {
      if (tokens[j] != blank && !(merge_repeated && tokens[j] == prev_token)) {
        output[ouput_idx] = tokens[j];
        ++ouput_idx;
      }
      pre_token = tokens[j];
    }
    out_lod0[i + 1] = ouput_idx;
  }
}

template <typename T>
class CTCGreedyDecodeOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "It must use CUDAPlace.");
    auto* input = ctx.Input<LoDTensor>("Input");
    auto* output = ctx.Output<LoDTensor>("Output");

    size_t num_tokens = input->dims()[0];
    size_t seq_width = input.numel() / num_tokens;
    const T* logits = input->data<T>();
    Tensor tmp;
    int* tokens = tmp.mutable_data<int>({num_tokens, 1}, ctx.GetPlace());
    // get argmax
    // platform::GpuMemsetAsync(args, 0, sizeof(float), stream);

    auto stream = ctx.cuda_device_context().stream();
    ArgmaxCudaKernel<T><<<num_tokens, PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
        seq_wisth, logits, tokens);

    const size_t level = 0;
    auto input_lod = framework::ToAbsOffset(input->lod());
    const size_t num_seq = input_lod[level].size() - 1;
    const int blank = ctx.Attr<int>("blank");
    const bool merge_repeated = ctx.Attr<int>("merge_repeated");

    thrust::device_vector<int> dev_out_lod0(input_lod[level].size());
    int* output_data =
        output->mutable_data<int>({num_tokens, 1}, ctx.GetPlace());
    MergeAndDelCudaKernel<T><<<1, 1, 0, stream>>>(
        num_tokens, tokens, num_seq, input_lod[level].data(), blank,
        merge_repeated, dev_out_lod0.data(), output_data);

    thrust::host_vector<int> host_out_lod0(dev_out_lod0.begin(),
                                           dev_out_lod0.end());
    framework::LoD out_lod;
    out_lod.push_back(host_out_lod0);
    output->set_lod(out_lod);

    output->Resize({host_out_lod0.back(), 1});
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_CUDA_KERNEL(ctc_greedy_decode,
                        paddle::operators::CTCGreedyDecodeOpCUDAKernel<float>);
