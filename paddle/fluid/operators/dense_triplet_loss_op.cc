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
#include "paddle/fluid/operators/dense_triplet_loss_op.h"

namespace paddle {
namespace operators {

template <>
std::vector<int> GetOffsets<platform::CPUDeviceContext>(const Tensor* t) {
  std::vector<int> offsets;
  offsets.push_back(0);
  const int64_t* data = t->data<int64_t>();
  int64_t currrent_value = data[0];
  for (int i = 1; i < t->numel(); ++i) {
    if (data[i] != currrent_value) {
      offsets.push_back(i);
    }
    currrent_value = data[i];
  }
  offsets.push_back(t->numel());
  return offsets;
}

class DenseTripletLossOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Logits",
             "(Tensor, default: Tensor<float>), A 2-D tensor with shape [N x "
             "K]. N is the batch_size, "
             "and K is the feature length in each sample.");
    AddInput("Label",
             "(Tensor) The ground truth which is a 2-D tensor.  "
             "Label is a Tensor<int64> with shape [N x 1]. ");
    AddOutput("Loss",
              "(Tensor, default: Tensor<float>), A 2-D tensor. The triplet "
              "loss with shape [batch_size x 1].");
    AddOutput("LogitsGrad",
              "(Tensor, default: Tensor<float>), A temporary "
              "output Tensor to store the gradients of triplet loss, which is "
              "computed with loss together in one call. It is a 2-D Tensor of "
              "the shape [batch_size, feature_len].")
        .AsIntermediate();
    AddAttr<float>("margin", "(float), The min margin between two sample.");

    AddComment(R"DOC(

)DOC");
  }
};

class DenseTripletLossOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Logits"),
                   "Input(Logits) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput("Label"), "Input(Label) should be not null.");
    PADDLE_ENFORCE(ctx->HasOutput("Loss"), "Output(Loss) should be not null.");
    PADDLE_ENFORCE(ctx->HasOutput("LogitsGrad"),
                   "Output(LogitsGrad) should be not null.");
    auto labels_dims = ctx->GetInputDim("Label");
    auto logits_dims = ctx->GetInputDim("Logits");
    PADDLE_ENFORCE_EQ(
        logits_dims.size(), 2UL,
        "The input of dense_triplet_loss should be a 2-D tensor.");
    PADDLE_ENFORCE_EQ(labels_dims.size(), 2UL,
                      "The labels should be a 2-D tensor.");
    PADDLE_ENFORCE_EQ(labels_dims[1], 1UL,
                      "The 2nd dimension of "
                      "Input(Label) should be 1.");
    ctx->SetOutputDim("Loss", {logits_dims[0], 1});
    ctx->SetOutputDim("LogitsGrad", logits_dims);
    ctx->ShareLoD("Logits", /*->*/ "Loss");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<Tensor>("Logits")->type()),
        ctx.device_context());
  }
};

class DenseTripletLossGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Loss")),
                   "Input(Loss@Grad) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Label"), "Input(Label) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput("LogitsGrad"),
                   "Input(LogitsGrad) should be not null.");
    PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("Logits")),
                   "Output(Logits@Grad) should be not null.");

    auto labels_dims = ctx->GetInputDim("Label");
    PADDLE_ENFORCE_EQ(labels_dims.size(), 2UL,
                      "The labels should be a 2-D tensor.");

    PADDLE_ENFORCE_EQ(labels_dims[1], 1UL,
                      "the 2nd dimension of Input(Label) should be 1.");

    ctx->SetOutputDim(framework::GradVarName("Logits"),
                      ctx->GetInputDim("LogitsGrad"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(
            ctx.Input<Tensor>(framework::GradVarName("Loss"))->type()),
        ctx.device_context());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(dense_triplet_loss, ops::DenseTripletLossOp,
                  ops::DenseTripletLossOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);

REGISTER_OPERATOR(dense_triplet_loss_grad, ops::DenseTripletLossGradOp);

REGISTER_OP_CPU_KERNEL(
    dense_triplet_loss,
    ops::DenseTripletLossKernel<paddle::platform::CPUDeviceContext, float>,
    ops::DenseTripletLossKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    dense_triplet_loss_grad,
    ops::DenseTripletLossGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::DenseTripletLossGradKernel<paddle::platform::CPUDeviceContext,
                                    double>);
