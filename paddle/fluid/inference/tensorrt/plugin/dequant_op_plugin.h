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

#pragma once

#include <algorithm>
#include <string>
#include <vector>
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/tensor_util.h"

#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

//void Float2Char(float* pValues, char* pDst, size_t nBit, size_t nCount) {
//  for (size_t i=0; i<nCount; i++) {
//      
//  }
//
//}

class DequantPlugin : public PluginTensorRTV2Ext {
  std::vector<char> weight_;
  char* p_gpu_weight_;
  size_t n_bit_;
  size_t n_weight_num_;
  nvinfer1::Dims weight_dims_;

 public:
  size_t getSerializationSize() const TRT_NOEXCEPT override {
    return getBaseSerializationSize() + SerializedSize(n_bit_) +
           SerializedSize(n_weight_num_) + SerializedSize(weight_) +
	   SerializedSize(weight_dims_);
  }

  // TRT will call this func when we need to serialize the configuration of
  // tensorrt.
  // It should not be called by users.
  void serialize(void* buffer) const TRT_NOEXCEPT override {
    serializeBase(buffer);
    SerializeValue(&buffer, weight_);
    SerializeValue(&buffer, n_bit_);
    SerializeValue(&buffer, n_weight_num_);
    SerializeValue(&buffer, weight_dims_);
  }

  DequantPlugin(const float* weight, const int weight_num, size_t n_bit, const nvinfer1::Dims weight_dims)
	  : n_bit_(n_bit), 
	  n_weight_num_(weight_num),
	  weight_dims_(weight_dims) {
    size_t char_num = weight_num * n_bit / 8;
    weight_.resize(char_num);
    //std::copy(weight, weight + weight_num, weight_.data());
  }

  // It was used for tensorrt deserialization.
  // It should not be called by users.
  DequantPlugin(void const* serialData, size_t serialLength) {
    deserializeBase(serialData, serialLength);
    DeserializeValue(&serialData, &serialLength, &weight_);
    DeserializeValue(&serialData, &serialLength, &n_bit_);
    DeserializeValue(&serialData, &serialLength, &n_weight_num_);
    DeserializeValue(&serialData, &serialLength, &weight_dims_);
  }
  ~DequantPlugin() {}
  int initialize() TRT_NOEXCEPT override;
  void destroy() TRT_NOEXCEPT override;

  DequantPlugin* clone() const TRT_NOEXCEPT override {
    auto* ptr =
        new DequantPlugin(nullptr, n_weight_num_, n_bit_, weight_dims_);
    ptr->p_gpu_weight_ = p_gpu_weight_;
    ptr->weight_ = weight_;
    return ptr;
  }

  const char* getPluginType() const TRT_NOEXCEPT override {
    return "dequant_plugin";
  }
  int getNbOutputs() const TRT_NOEXCEPT override { return 1; }


  nvinfer1::DataType getOutputDataType(
      int index, const nvinfer1::DataType* input_types,
      int nb_inputs) const TRT_NOEXCEPT override;

  nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs,
                                     int nbInputDims) TRT_NOEXCEPT override;
#if IS_TRT_VERSION_LT(8000)
  int enqueue(int batchSize, const void* const* inputs, void** outputs,
#else
  int enqueue(int batchSize, const void* const* inputs, void* const* outputs,
#endif
              void* workspace, cudaStream_t stream) TRT_NOEXCEPT override;
};

class DequantPluginCreator : public TensorRTPluginCreator {
 public:
  const char* getPluginName() const TRT_NOEXCEPT override {
    return "dequant_plugin";
  }

  const char* getPluginVersion() const TRT_NOEXCEPT override { return "1"; }

  nvinfer1::IPluginV2* deserializePlugin(
      const char* name, const void* serial_data,
      size_t serial_length) TRT_NOEXCEPT override {
    return new DequantPlugin(serial_data, serial_length);
  }
};
REGISTER_TRT_PLUGIN_V2(DequantPluginCreator);

#if IS_TRT_VERSION_GE(6000)

class DequantPluginDynamic : public DynamicPluginTensorRT {
 public:
  DequantPluginDynamic(const float* weight, size_t weightNum, size_t nBit, const nvinfer1::Dims weightDims, float* scales, size_t nScales, int quantStride);

  DequantPluginDynamic(void const* serialData, size_t serialLength);
  DequantPluginDynamic() {}
  ~DequantPluginDynamic() {terminate();}

  nvinfer1::IPluginV2DynamicExt* clone() const TRT_NOEXCEPT override;

  const char* getPluginType() const TRT_NOEXCEPT override {
    return "dequant_plugin_dynamic";
  }
  int getNbOutputs() const TRT_NOEXCEPT override { return 1; }
  int initialize() TRT_NOEXCEPT override;
  void terminate() TRT_NOEXCEPT override;

  size_t getSerializationSize() const TRT_NOEXCEPT override;
  void serialize(void* buffer) const TRT_NOEXCEPT override;

  nvinfer1::DimsExprs getOutputDimensions(
      int output_index, const nvinfer1::DimsExprs* inputs, int nb_inputs,
      nvinfer1::IExprBuilder& expr_builder) TRT_NOEXCEPT override;

  bool supportsFormatCombination(int pos,
                                 const nvinfer1::PluginTensorDesc* inOut,
                                 int nbInputs,
                                 int nbOutputs) TRT_NOEXCEPT override;

  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                       int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc* out,
                       int nbOutputs) TRT_NOEXCEPT override {}

  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                          int nbInputs,
                          const nvinfer1::PluginTensorDesc* outputs,
                          int nbOutputs) const TRT_NOEXCEPT override {
    return 0;
  }

  int enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
              const nvinfer1::PluginTensorDesc* outputDesc,
              const void* const* inputs, void* const* outputs, void* workspace,
              cudaStream_t stream) TRT_NOEXCEPT override;
  nvinfer1::DataType getOutputDataType(
      int index, const nvinfer1::DataType* inputTypes,
      int nbInputs) const TRT_NOEXCEPT override;

  void destroy() TRT_NOEXCEPT override { delete this; }

 private:
  size_t n_bit_;
  nvinfer1::Dims weight_dims_;
  std::vector<float> scales_;
  int quant_stride_;

  std::vector<int8_t> weight_;
  int8_t* p_gpu_weight_;
  float* p_gpu_scales_;

};
#endif

class DequantPluginDynamicCreator : public TensorRTPluginCreator {
 public:
  const char* getPluginName() const TRT_NOEXCEPT override {
    return "dequant_plugin_dynamic";
  }

  const char* getPluginVersion() const TRT_NOEXCEPT override { return "1"; }

  nvinfer1::IPluginV2* deserializePlugin(
      const char* name, const void* serial_data,
      size_t serial_length) TRT_NOEXCEPT override {
    return new DequantPluginDynamic(serial_data, serial_length);
  }
};
REGISTER_TRT_PLUGIN_V2(DequantPluginDynamicCreator);

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
