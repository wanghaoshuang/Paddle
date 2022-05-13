// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdio.h>

#include <cassert>
#include <vector>

#include "glog/logging.h"
#include "paddle/fluid/inference/tensorrt/plugin/dequant_op_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

int DequantPlugin::initialize() TRT_NOEXCEPT {
  cudaMalloc(&p_gpu_weight_, sizeof(char) * weight_.size());
  cudaMemcpy(p_gpu_weight_, weight_.data(), weight_.size() * sizeof(char),
             cudaMemcpyHostToDevice);
  return 0;
}

void DequantPlugin::destroy() TRT_NOEXCEPT {
  if (p_gpu_weight_) {
    cudaFree(p_gpu_weight_);
    p_gpu_weight_ = nullptr;
  }
}


nvinfer1::DataType DequantPlugin::getOutputDataType(
      int index, const nvinfer1::DataType* input_types,
      int nb_inputs) const TRT_NOEXCEPT {
  VLOG(3) << "DequantPlugin::getOutputDataType---";
  return nvinfer1::DataType::kHALF;
}

nvinfer1::Dims DequantPlugin::getOutputDimensions(int index,
                                                const nvinfer1::Dims *inputDims,
                                                int nbInputs) TRT_NOEXCEPT {
  VLOG(3) << "DequantPlugin::getOutputDimensions: " << weight_dims_.nbDims;
  assert(index < this->getNbOutputs());
  return weight_dims_;
}

int DequantPlugin::enqueue(int batch_size, const void *const *inputs,
#if IS_TRT_VERSION_LT(8000)
                         void **outputs, void *workspace, cudaStream_t stream) {
#else
                         void *const *outputs, void *workspace,
                         cudaStream_t stream) TRT_NOEXCEPT {
#endif
  return true;
}



#if IS_TRT_VERSION_GE(6000)
inline void encode_int4(int8_t a, int8_t b, int8_t& n) {
  int8_t l = (a & 0x07) | (0x08 & (a>>7<<3));
  int8_t h = (b & 0x80) | ((b << 4) & 0x70);
  n = (h | l);
}

void encode_all_int4(const float* src, size_t n_src, int8_t* target) {
  for (size_t i = 0, j=0; i< n_src; i+=2, j++) {
    encode_int4(static_cast<int8_t>(src[i]), static_cast<int8_t>(src[i+1]), target[j]);
  }
}

DequantPluginDynamic::DequantPluginDynamic(const float* weight, size_t weightNum, size_t nBit, const nvinfer1::Dims weightDims, float* scales, size_t nScales, int quant_stride)
    : n_bit_(nBit), 
    weight_dims_(weightDims),
          quant_stride_(quant_stride) {
  size_t n_encoded = (weightNum * n_bit_) / 8;
  weight_.resize(n_encoded);

  if (n_bit_ == 8) {
    for (size_t i = 0; i < weightNum; i++) {
      weight_[i] = static_cast<int8_t>(weight[i]);
    }
  } else if (n_bit_ == 4) {
    encode_all_int4(weight, weightNum, weight_.data());
  } else {
  
  }
  scales_.resize(nScales); 
  for (size_t i=0; i<nScales; i++) {
    scales_[i] = scales[i];
  }
}


nvinfer1::IPluginV2DynamicExt* DequantPluginDynamic::clone() const TRT_NOEXCEPT {
  VLOG(3) << "DequantPluginDynamic::clone()";
  auto* ptr = new DequantPluginDynamic();
  ptr->p_gpu_weight_ = p_gpu_weight_;
  ptr->p_gpu_scales_ = p_gpu_scales_;
  ptr->weight_.assign( weight_.begin(), weight_.end());
  
  ptr->n_bit_ = n_bit_;
  ptr->weight_dims_.nbDims = weight_dims_.nbDims;
  for (size_t i=0;i<weight_dims_.nbDims; i++) {
    ptr->weight_dims_.d[i] =  weight_dims_.d[i];
  }
  ptr->scales_.assign(scales_.begin(), scales_.end());
  ptr->quant_stride_ = quant_stride_;
  return ptr;
}



void DequantPluginDynamic::terminate() TRT_NOEXCEPT {
  VLOG(3) << "DequantPluginDynamic::terminate()";
  
  if (p_gpu_weight_) {
    cudaFree(p_gpu_weight_);
    p_gpu_weight_ = nullptr;
  }
  if (p_gpu_scales_) {
    cudaFree(p_gpu_scales_);
    p_gpu_scales_ = nullptr;
  }
}

int DequantPluginDynamic::initialize() TRT_NOEXCEPT {
  VLOG(3) << "DequantPluginDynamic::initialize()";
  cudaMalloc(&p_gpu_weight_, sizeof(int8_t) * weight_.size());
  cudaMemcpy(p_gpu_weight_, weight_.data(), weight_.size() * sizeof(int8_t),
             cudaMemcpyHostToDevice);

  cudaMalloc(&p_gpu_scales_, sizeof(float) * scales_.size());
  cudaMemcpy(p_gpu_scales_, scales_.data(), sizeof(float) * scales_.size(),
             cudaMemcpyHostToDevice);

  return 0;
}

DequantPluginDynamic::DequantPluginDynamic(void const *serialData,
                                       size_t serialLength) {
//    deserializeBase(serialData, serialLength);
    DeserializeValue(&serialData, &serialLength, &weight_);
    DeserializeValue(&serialData, &serialLength, &n_bit_);
    DeserializeValue(&serialData, &serialLength, &weight_dims_);
    DeserializeValue(&serialData, &serialLength, &scales_);
    DeserializeValue(&serialData, &serialLength, &quant_stride_);
}

size_t DequantPluginDynamic::getSerializationSize() const TRT_NOEXCEPT {
  return SerializedSize(n_bit_) + SerializedSize(weight_) +
	   SerializedSize(weight_dims_) + SerializedSize(scales_)+
          SerializedSize(quant_stride_);
}


void DequantPluginDynamic::serialize(void *buffer) const TRT_NOEXCEPT {
//    serializeBase(buffer);
    SerializeValue(&buffer, weight_);
    SerializeValue(&buffer, n_bit_);
    SerializeValue(&buffer, weight_dims_);
    SerializeValue(&buffer, scales_);
    SerializeValue(&buffer, quant_stride_);
}

nvinfer1::DimsExprs DequantPluginDynamic::getOutputDimensions(
    int output_index, const nvinfer1::DimsExprs *inputs, int nb_inputs,
    nvinfer1::IExprBuilder &expr_builder) TRT_NOEXCEPT {
  assert(index < this->getNbOutputs());
  nvinfer1::DimsExprs ret;
  ret.nbDims = weight_dims_.nbDims + 1;
  ret.d[0] = expr_builder.constant(1);
  for(int i = 1; i<ret.nbDims; i++) {
    ret.d[i] = expr_builder.constant(weight_dims_.d[i-1]);
  }
  return ret;
}

bool DequantPluginDynamic::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc *in_out, int nb_inputs,
    int nb_outputs) TRT_NOEXCEPT {
  PADDLE_ENFORCE_NOT_NULL(
      in_out, platform::errors::InvalidArgument(
                  "The input of dequant plugin shoule not be nullptr."));

  PADDLE_ENFORCE_LT(
      pos, nb_inputs + nb_outputs,
      platform::errors::InvalidArgument("The pos(%d) should be less than the "
                                        "num(%d) of the input and the output.",
                                        pos, nb_inputs + nb_outputs));
  (in_out && pos < (nb_inputs + nb_outputs));

  return ((in_out[pos].type == nvinfer1::DataType::kFLOAT) || (in_out[pos].type == nvinfer1::DataType::kHALF) &&
          in_out[pos].format == nvinfer1::PluginFormat::kLINEAR);
}

nvinfer1::DataType DequantPluginDynamic::getOutputDataType(
    int index, const nvinfer1::DataType *input_types,
    int nb_inputs) const TRT_NOEXCEPT {
  PADDLE_ENFORCE_EQ(index, 0, platform::errors::InvalidArgument(
                                  "The Dequant Plugin only has one output, so the "
                                  "index value should be 0, but get %d.",
                                  index));
  PADDLE_ENFORCE_EQ((input_types[0] == nvinfer1::DataType::kFLOAT), true,
                    platform::errors::InvalidArgument(
                        "The input type should be half or float"));
  return nvinfer1::DataType::kHALF;
}


//template <typename T>
//__global__ void dequant_from_4bits(int8_t* input, T* output, int nbInput, float* scales, int quant_stride) {
//  const int idx  = blockIdx.x * blockDim.x + threadIdx.x;
//  if (idx < nbInput) {
//    int8_t n =  input[idx];
//   int8_t low = (n << 4) >> 4;
//   int8_t high = n >> 4;
//   size_t out_idx = 2 * idx;
//   output[out_idx] = static_cast<T>(static_cast<float>(low) * scales[out_idx / quant_stride]);
//   output[out_idx+1] = static_cast<T>(static_cast<float>(high) * scales[(out_idx+1)/quant_stride]);
//  }
//}  


template <typename T>
__global__ void dequant_from_4bits(int8_t* input, T* output, int nbInput, float* scales, int quant_stride) {
  const int idx  = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < nbInput) {
    int8_t n =  input[idx];
   int8_t low = (n << 4) >> 4;
   int8_t high = n >> 4;
   size_t out_idx = 2 * idx;
   output[out_idx] = static_cast<T>(static_cast<float>(low) * scales[out_idx / quant_stride]);
   output[out_idx+1] = static_cast<T>(static_cast<float>(high) * scales[(out_idx+1)/quant_stride]);
  }
}  

template <typename T>
__global__ void dequant_from_8bits(int8_t* input, T* output, int nbInput, float* scales, int quant_stride) {
  const int idx  = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < nbInput) {
    //int8_t n =  input[idx];
    // output[idx] = static_cast<T>(static_cast<float>(n) * scales[idx / quant_stride]);
//    output[idx] = static_cast<T>(static_cast<float>(n));
  }
}  



int DequantPluginDynamic::enqueue(const nvinfer1::PluginTensorDesc *input_desc,
                                const nvinfer1::PluginTensorDesc *output_desc,
                                const void *const *inputs, void *const *outputs,
                                void *workspace,
                                cudaStream_t stream) TRT_NOEXCEPT {
  auto output_type = output_desc[0].type;
  int threads = 1024;
  int blocks = ((weight_.size() + threads - 1) / threads)/8;
//  VLOG(3) << "scales_.size(): " <<  scales_.size() << "; quant_stride_:" << quant_stride_ << "; weight._size():"<< weight_.size();
  auto out_dims = output_desc[0].dims;
//  VLOG(3) << "out dims nbDims: " << out_dims.nbDims << "; [" << out_dims.d[0] << ", " << out_dims.d[1] << ", " << out_dims.d[2] << ", " << out_dims.d[3] << "]";

//  PADDLE_ENFORCE_NOT_NULL(
//      p_gpu_weight_, platform::errors::InvalidArgument(
//                  "p_gpu_weight_ shoule not be nullptr."));
//
//  PADDLE_ENFORCE_NOT_NULL(
//      p_gpu_scales_, platform::errors::InvalidArgument(
//                  "p_gpu_scales_ shoule not be nullptr."));
  if (n_bit_ == 4) {
    if (output_type == nvinfer1::DataType::kHALF) {
      dequant_from_4bits<half><<<blocks, threads, 0, stream>>>(p_gpu_weight_,
                                                              static_cast<half*>(outputs[0]),
                                                              weight_.size(),
                                                              p_gpu_scales_,
                                                              quant_stride_);
    //  cudaStreamSynchronize(stream);
    } else if (output_type == nvinfer1::DataType::kFLOAT) {
      dequant_from_4bits<float><<<blocks, threads, 0, stream>>>(p_gpu_weight_,
                                                              static_cast<float*>(outputs[0]),
                                                              weight_.size(),
                                                              p_gpu_scales_,
                                                              quant_stride_);
//      cudaStreamSynchronize(stream);
    } 
  } else if (n_bit_ == 8){
     if (output_type == nvinfer1::DataType::kHALF) {
      dequant_from_8bits<half><<<blocks, threads, 0, stream>>>(p_gpu_weight_,
                                                              static_cast<half*>(outputs[0]),
                                                              weight_.size(),
                                                              p_gpu_scales_,
                                                              quant_stride_);
    //  cudaStreamSynchronize(stream);
    } else if (output_type == nvinfer1::DataType::kFLOAT) {
      dequant_from_8bits<float><<<blocks, threads, 0, stream>>>(p_gpu_weight_,
                                                              static_cast<float*>(outputs[0]),
                                                              weight_.size(),
                                                              p_gpu_scales_,
                                                              quant_stride_);
//      cudaStreamSynchronize(stream);
    } 
  }
  return cudaGetLastError() != cudaSuccess;
}
#endif

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
