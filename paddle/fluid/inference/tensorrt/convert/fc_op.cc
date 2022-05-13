/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
#include "paddle/fluid/inference/tensorrt/plugin/matmul_op_int8_plugin.h"
#include "paddle/fluid/inference/tensorrt/plugin/dequant_op_plugin.h"

namespace paddle {
namespace framework {
class Scope;

namespace proto {
class OpDesc;
}  // namespace proto
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace inference {
namespace tensorrt {

/*
 * FC converter convert a MUL op in Fluid to a FC layer in TRT.
 */
class FcOpConverter : public OpConverter {
 public:
  nvinfer1::ILayer* reshape_before_fc(nvinfer1::ITensor* before_fc,
                                      nvinfer1::Dims x_dim, int x_num_col_dims,
                                      std::string output_name) {
    // add shuffle before fc
    nvinfer1::Dims reshape_before_fc_dim;
    reshape_before_fc_dim.nbDims = x_num_col_dims + 3;
    // padding shape "* x q x 1 x 1"
    for (int i = 0; i < reshape_before_fc_dim.nbDims; i++) {
      reshape_before_fc_dim.d[i] = 1;
    }
    for (int i = 0; i < x_dim.nbDims; i++) {
      if (i < x_num_col_dims) {
        reshape_before_fc_dim.d[i] = 0;
      } else {
        if (x_dim.d[i] < 0) {
          reshape_before_fc_dim.d[x_num_col_dims] = -1;
          break;
        }
        reshape_before_fc_dim.d[x_num_col_dims] *= x_dim.d[i];
      }
    }
    auto* reshape_before_fc_layer =
        TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *before_fc);
    reshape_before_fc_layer->setReshapeDimensions(reshape_before_fc_dim);
    VLOG(3) << "reshape_before_fc_dim, nbDims: " << reshape_before_fc_dim.nbDims << "; [" << reshape_before_fc_dim.d[0] << ", " << reshape_before_fc_dim.d[1] << "," << reshape_before_fc_dim.d[2] << "];";
    reshape_before_fc_layer->setName(
        ("fc_op_reshape_before_fc: Shuffle (Output: " + output_name + ")")
            .c_str());
    return reshape_before_fc_layer;
  }

  nvinfer1::ILayer* reshape_after_fc(nvinfer1::ITensor* after_fc,
                                     nvinfer1::Dims x_dim, int x_num_col_dims) {
    // add shuffle after fc
    nvinfer1::Dims reshape_after_fc_dim;
    reshape_after_fc_dim.nbDims = x_num_col_dims + 1;
    for (int i = 0; i < reshape_after_fc_dim.nbDims; i++) {
      reshape_after_fc_dim.d[i] = 0;
    }
    auto* reshape_after_fc_layer =
        TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *after_fc);
    reshape_after_fc_layer->setReshapeDimensions(reshape_after_fc_dim);
    VLOG(3) << "reshape_after_fc_dim, nbDims: " << reshape_after_fc_dim.nbDims << "; [" << reshape_after_fc_dim.d[0] << ", " << reshape_after_fc_dim.d[1] << "," << reshape_after_fc_dim.d[2] << "];";
    return reshape_after_fc_layer;
  }

  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(3) << "convert a fluid fc op to tensorrt fc layer without bias";
    framework::OpDesc op_desc(op, nullptr);
    auto output_name = op_desc.Output("Out").front();
    VLOG(3) << "output of fc is : " << output_name;
    auto input_names = op_desc.InputNames();
    bool with_bias = input_names.size() >= 3;
    std::string w_name = "Y";
    std::string i_name = "X";
    if (with_bias) {
      w_name = "W";
      i_name = "Input";
    }
    // Declare inputs
    auto* X = engine_->GetITensor(op_desc.Input(i_name).front());
    auto x_dim = X->getDimensions();
    // Declare weights
    auto* Y_v = scope.FindVar(op_desc.Input(w_name).front());
    PADDLE_ENFORCE_NOT_NULL(
        Y_v, platform::errors::NotFound(
                 "Can not find %s presistale var of fc in scope.", w_name));
    auto* Y_t = Y_v->GetMutable<framework::LoDTensor>();
    int x_num_col_dims =
        op_desc.HasAttr("x_num_col_dims")
            ? BOOST_GET_CONST(int, op_desc.GetAttr("x_num_col_dims"))
            : (op_desc.HasAttr("in_num_col_dims")
                   ? BOOST_GET_CONST(int, op_desc.GetAttr("in_num_col_dims"))
                   : 1);
    const std::string activation_type =
        op_desc.HasAttr("activation_type")
            ? BOOST_GET_CONST(std::string, op_desc.GetAttr("activation_type"))
            : "";
    // This may trigger a GPU->CPU copy, because TRT's weight can only be
    // assigned from CPU memory, which can't be avoided.
    float* weight_data = nullptr;
    bool enable_int8 = op_desc.HasAttr("enable_int8");

    float in_scale = 0.;
    int quantize_weight_bits = 0;

    if (op_desc.HasAttr("quantize_weight_bits")) {
      quantize_weight_bits = BOOST_GET_CONST(int, op_desc.GetAttr("quantize_weight_bits"));
    }
    VLOG(3) << "enable_int8: " << enable_int8 << "; quantize_weight_bits: " << quantize_weight_bits; 
    if (enable_int8) {
#if IS_TRT_VERSION_GE(5000)
      auto weight_scale =
          BOOST_GET_CONST(std::vector<float>, op_desc.GetAttr("weight_scale"));
      weight_data = engine_->GetWeightCPUData(op_desc.Input(w_name).front(),
                                              Y_t, true, weight_scale);

      in_scale =
            BOOST_GET_CONST(float, op_desc.GetAttr(i_name + "_scale")) * 127;
      engine_->SetTensorDynamicRange(X, in_scale);
#endif
    } else {
      weight_data =
          engine_->GetWeightCPUData(op_desc.Input(w_name).front(), Y_t, false);
    }

    PADDLE_ENFORCE_EQ(Y_t->dims().size(), 2UL,
                      platform::errors::InvalidArgument(
                          "The fc's weight should be a matrix with 2 dims, but "
                          "it's %d-dimensional.",
                          Y_t->dims().size()));  // a matrix
    int m = Y_t->dims()[0]; // hidden_in
    int n = Y_t->dims()[1]; // hidden_out
    auto tranpose_weight = [](const float* src, float* dst, int m, int n) {
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
          dst[j * m + i] = src[i * n + j];
        }
      }
    };

    auto regist_fc = [&](nvinfer1::ITensor* inputs, int n_output,
                         float* weight_data,
			 TensorRTEngine::Weight& weight,
                         TensorRTEngine::Weight& bias) {
      VLOG(4) << "regist_fc; enable_int8: " << enable_int8;
      if (enable_int8 && (quantize_weight_bits==0)) { // both weights and activations are quantized.
        // add conv layer
        PADDLE_ENFORCE_EQ(
            op_desc.HasAttr("out_threshold"), true,
            platform::errors::InvalidArgument(
                "must have out threshold in fc layers in int8 mode"));
        float out_scale =
            BOOST_GET_CONST(float, op_desc.GetAttr("out_threshold"));
        nvinfer1::DimsHW nv_ksize(1, 1);
        auto* fc_layer_int8 =
            TRT_ENGINE_ADD_LAYER(engine_, Convolution, *inputs, n_output,
                                 nv_ksize, weight.get(), bias.get());
        fc_layer_int8->setName(
            ("fc_op_int8_conv1x1: Convolution (Output: " + output_name + ")")
                .c_str());
        engine_->SetTensorDynamicRange(fc_layer_int8->getOutput(0), out_scale);
        auto* fc_after_reshape_int8 = reshape_after_fc(
            fc_layer_int8->getOutput(0), x_dim, x_num_col_dims);
        if (activation_type == "relu") {
          fc_after_reshape_int8->setName(
              ("int8_reshape_after_fc: Shuffle (Output: " + output_name + ")")
                  .c_str());
          engine_->SetTensorDynamicRange(fc_after_reshape_int8->getOutput(0),
                                         out_scale);
          nvinfer1::IActivationLayer* relu_layer_int8 = TRT_ENGINE_ADD_LAYER(
              engine_, Activation, *(fc_after_reshape_int8->getOutput(0)),
              nvinfer1::ActivationType::kRELU);
          RreplenishLayerAndOutput(relu_layer_int8, "relu_after_fc_shuffle",
                                   {output_name}, test_mode);
        } else {
          RreplenishLayerAndOutput(fc_after_reshape_int8,
                                   "fc_op_int8_reshape_after_fc: Shuffle",
                                   {output_name}, test_mode);
        }
      } else if (quantize_weight_bits!=0) { // just quantized weights
	nvinfer1::Dims weight_dims;
	weight_dims.nbDims = 2;
	weight_dims.d[0] = weight.dims[0];
	weight_dims.d[1] = weight.dims[1];

       
        auto scale_name  = op_desc.HasAttr("weight_scale") ? "weight_scale" : op_desc.Input(w_name).front() + "_quant_scale"; 
        auto weight_scale =
          BOOST_GET_CONST(std::vector<float>, op_desc.GetAttr(scale_name));
         
        int quant_stride = weight_scale.size() > 1 ? weight_dims.d[1] : weight_dims.d[0] * weight_dims.d[1];;

        // Add dequant
	VLOG(5) << "Add DequantPluginDynamic to dequant weights from integer to float just in time.";
        plugin::DynamicPluginTensorRT* plugin =
            new plugin::DequantPluginDynamic(weight_data, weight.get().count, quantize_weight_bits, weight_dims,
                                            weight_scale.data(), weight_scale.size(), quant_stride);
        
        // Adapt interface of dynamic plugin by a dummy input "X"
        auto *dequant_layer = engine_->AddDynamicPlugin(&X, 1, plugin); 

	auto *dequant_out = dequant_layer->getOutput(0);
	auto dequant_dims = dequant_out->getDimensions();
	VLOG(3) << "out of dequant, nbDims: " << dequant_dims.nbDims << "; [" << dequant_dims.d[0] << ", " << dequant_dims.d[1] << ", " << dequant_dims.d[2] << "];";

	// Add matmul
	std::vector<nvinfer1::ITensor*> matmul_inputs;
        matmul_inputs.push_back(inputs);
        matmul_inputs.push_back(dequant_layer->getOutput(0));
        auto *fc_layer_float = TRT_ENGINE_ADD_LAYER(engine_, MatrixMultiply, *inputs, nvinfer1::MatrixOperation::kNONE, *(dequant_layer->getOutput(0)), nvinfer1::MatrixOperation::kTRANSPOSE);

	auto *out = fc_layer_float->getOutput(0);
	auto dims = out->getDimensions();
	VLOG(3) << "out of matmul, nbDims: " << dims.nbDims << "; [" << dims.d[0] << ", " << dims.d[1] <<","<< dims.d[2] <<"];";

        if (activation_type == "relu") {
          fc_layer_float->setName(
            ("fc_op_float: MatrixMultiply (Output: " + output_name + ")")
                .c_str());
          nvinfer1::IActivationLayer* relu_layer_float = TRT_ENGINE_ADD_LAYER(
              engine_, Activation, *(fc_layer_float->getOutput(0)),
              nvinfer1::ActivationType::kRELU);
          RreplenishLayerAndOutput(relu_layer_float, "relu_after_fc",
                                   {output_name}, test_mode);
        } else {
          VLOG(3) << "RreplenishLayerAndOutput, output_name: " << output_name;
          RreplenishLayerAndOutput(fc_layer_float, "fc_layer_float",
                                   {output_name}, test_mode);
        }
      } else {

      }
    };

    bool transpose_y = false;
    if (op_desc.HasAttr("transpose_Y")) {
      transpose_y = BOOST_GET_CONST(bool, op_desc.GetAttr("transpose_Y"));
    }
    int weight_w, weight_h;
    if (!transpose_y) {
      std::vector<float> weight_data_tmp;
      weight_data_tmp.reserve(Y_t->numel());
      memcpy(weight_data_tmp.data(), weight_data, Y_t->numel() * sizeof(float));
      tranpose_weight(weight_data_tmp.data(), weight_data, m, n);
      weight_w = n;
      weight_h = m;
    } else {
      weight_w = m;
      weight_h = n;
    }
    size_t n_output = weight_w;
    TensorRTEngine::Weight weight{nvinfer1::DataType::kFLOAT,
                                  static_cast<void*>(weight_data),
                                  static_cast<size_t>(Y_t->numel())};
    weight.dims.assign({weight_w, weight_h});

    float* bias_data = nullptr;
    int bias_num = 0;
    if (with_bias) {
      auto* b_v = scope.GetVar(op_desc.Input("Bias").front());
      auto* b_t = b_v->GetMutable<framework::LoDTensor>();
      bias_data =
          engine_->GetWeightCPUData(op_desc.Input("Bias").front(), b_t, false);
      bias_num = b_t->numel();
    }
    TensorRTEngine::Weight bias{nvinfer1::DataType::kFLOAT,
                                static_cast<void*>(bias_data),
                                static_cast<size_t>(bias_num)};

    // Running the TRT Static Shape mode: x_num_col_dims-1
    if (!engine_->with_dynamic_shape()) {
      x_num_col_dims--;
    }

    PADDLE_ENFORCE_EQ(
      engine_->use_oss() && (quantize_weight_bits!=0), false,
      platform::errors::InvalidArgument(
        "Unsupport models just quantized weights in tensorrt oss mode."));

    // If use tensorrt'oss, the x_dim and x_num_col_dims need change, and can
    // not add Shuffle layer in ernie's multihead.
    if (engine_->use_oss() && engine_->with_ernie() && x_dim.nbDims == 4 &&
        x_dim.d[3] == 1 && x_num_col_dims == 2) {
      if (enable_int8) {
        // add conv1x1 layer
        nvinfer1::DimsHW nv_ksize(1, 1);
        auto* fc_layer_int8 =
            TRT_ENGINE_ADD_LAYER(engine_, Convolution, *X, n_output, nv_ksize,
                                 weight.get(), bias.get());
        if (activation_type == "relu") {
          fc_layer_int8->setName(
              ("ernie_fc_op_int8: Convolution (Output: " + output_name + ")")
                  .c_str());
          PADDLE_ENFORCE_EQ(
              op_desc.HasAttr("out_threshold"), true,
              platform::errors::InvalidArgument(
                  "must have out threshold in fc layers in int8 mode"));
          float out_scale =
              BOOST_GET_CONST(float, op_desc.GetAttr("out_threshold"));
          engine_->SetTensorDynamicRange(fc_layer_int8->getOutput(0),
                                         out_scale);
          nvinfer1::IActivationLayer* relu_layer_int8 = TRT_ENGINE_ADD_LAYER(
              engine_, Activation, *(fc_layer_int8->getOutput(0)),
              nvinfer1::ActivationType::kRELU);
          RreplenishLayerAndOutput(relu_layer_int8, "relu_after_ernie_fc_int8",
                                   {output_name}, test_mode);
        } else {
          RreplenishLayerAndOutput(fc_layer_int8,
                                   "ernie_fc_op_int8: Convolution",
                                   {output_name}, test_mode);
        }
      } else {
        // add fc layer
        auto* fc_layer_float = TRT_ENGINE_ADD_LAYER(
            engine_, FullyConnected, *X, n_output, weight.get(), bias.get());
        if (activation_type == "relu") {
          fc_layer_float->setName(
              ("ernie_fc_op_float: (Output: " + output_name + ")").c_str());
          nvinfer1::IActivationLayer* relu_layer_float = TRT_ENGINE_ADD_LAYER(
              engine_, Activation, *(fc_layer_float->getOutput(0)),
              nvinfer1::ActivationType::kRELU);
          RreplenishLayerAndOutput(relu_layer_float,
                                   "relu_after_ernie_fc_float", {output_name},
                                   test_mode);
        } else {
          RreplenishLayerAndOutput(fc_layer_float, "ernie_fc_op_float",
                                   {output_name}, test_mode);
        }
      }
    } else {  // need reshape input before and after fc
      PADDLE_ENFORCE_GT(
          x_dim.nbDims, x_num_col_dims,
          platform::errors::InvalidArgument(
              "Params and input dims mismatch. Paddle-TRT FC "
              "converter expects x_dim.nbDims > x_num_col_dims, but "
              "x_dim.nbDims : %d, x_num_col_dims : %d.",
              x_dim.nbDims, x_num_col_dims));
      VLOG(3) << "x_num_col_dims: " << x_num_col_dims;
      VLOG(3) << "x dims, nbDims: " << x_dim.nbDims << ", [" << x_dim.d[0] << "," << x_dim.d[1] << "," <<  x_dim.d[2] << "];";
      if (quantize_weight_bits == 0) {
        auto* reshape_before_fc_layer =
            reshape_after_fc(X, x_dim, x_num_col_dims); // debuggggggggggggggg
        auto* reshape_itensor = reshape_before_fc_layer->getOutput(0);
        if (enable_int8) {
          engine_->SetTensorDynamicRange(reshape_itensor, in_scale);
        }
        auto tmp_dim = reshape_itensor->getDimensions();
        VLOG(3) << "dims after first reshape, nbDims: " << tmp_dim.nbDims << ", [" << tmp_dim.d[0] << "," << tmp_dim.d[1] << "," <<  tmp_dim.d[2] << "];";
        regist_fc(reshape_itensor, n_output, weight_data, weight, bias);
      } else {
        regist_fc(X, n_output, weight_data, weight, bias);
      }
    }
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(fc, FcOpConverter);
