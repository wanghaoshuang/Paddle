/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <algorithm>
#include <string>
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace operators {

class QuantizeLinearOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X",
                   "QuantizeLinear");
    OP_INOUT_CHECK(ctx->HasOutput("Y"), "Output", "Y",
                   "QuantizeLinear");
    ctx->SetOutputDim("Y", ctx->GetInputDim("X"));
    ctx->ShareLoD("X", /*->*/ "Y");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }
};

class QuantizeLinearOpMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) Input is float data type.");
    AddInput("Scale", "(Tensor) Input is float data type.");
    AddInput("ZeroPoint", "(Tensor) Input is float data type.");
    AddOutput("Y",
              "(Tensor) Output of quantized low level tensor, "
              "but also saved as float data type.");
    AddAttr<int>("axis",
                 "(int, default 0) The axis for quantization. "
                 "For conv2d, depthwise_conv2d, conv2d_transpose "
                 "and mul, the quant_axis is equal to the cout axis.")
        .SetDefault(0)
        .AddCustomChecker([](const int& quant_axis) {
          PADDLE_ENFORCE_EQ(quant_axis == 0 || quant_axis == 1, true,
                            platform::errors::InvalidArgument(
                                "'quant_axis' should be 0 or 1, but "
                                "the received is %d",
                                quant_axis));
        });
    AddAttr<int>("bit_length", "(int, default 8)")
        .SetDefault(8)
        .AddCustomChecker([](const int& bit_length) {
          PADDLE_ENFORCE_EQ(bit_length >= 1 && bit_length <= 16, true,
                            platform::errors::InvalidArgument(
                                "'bit_length' should be between 1 and 16, but "
                                "the received is %d",
                                bit_length));
        });
    AddComment(R"DOC(
The scale of QuantizeLinear operator is a vector.
In detail, each channel of the input X has a scale value.

$$scale_c = max(abs(X_c))$$
$$range = 2^{bit\_length - 1} - 1$$
$$Out_c = round(\frac{X_c * range} {scale_c})$$
In above three formulas, the range value of c is as follow:
$$0 \leq c \lt \ the\ channel\ number\ of\ X$$
)DOC");
  }
};


}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CPU = paddle::platform::CPUDeviceContext;

REGISTER_OPERATOR(
    quantize_linear, ops::QuantizeLinearOp,
    ops::QuantizeLinearOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
//REGISTER_OP_CPU_KERNEL(quantize_linear,
//                       ops::QuantizeLinearKernel<CPU, float>);


REGISTER_OPERATOR(
    dequantize_linear, ops::QuantizeLinearOp,
    ops::QuantizeLinearOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);

