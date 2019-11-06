/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/collective/c_allreduce_op.h"

namespace paddle {
namespace operators {

template <typename T>
class CAllReduceSumOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  std::unique_ptr<T> Apply() const override {
    std::unique_ptr<T> retv(new T());
    retv->SetType("c_allreduce_sum");
    retv->SetInput("X", this->OutputGrad("Out"));
    retv->SetOutput("Out", this->InputGrad("X"));
    retv->SetAttrMap(this->Attrs());
    return retv;
  }
};

class CAllReduceSumOpMaker : public CAllReduceOpMaker {
 protected:
  std::string GetName() const override { return "Sum"; }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OPERATOR(c_allreduce_sum, ops::CAllReduceOp,
                  ops::CAllReduceSumOpGradMaker<paddle::framework::OpDesc>,
                  ops::CAllReduceSumOpGradMaker<paddle::imperative::OpBase>,
                  ops::CAllReduceSumOpMaker);

REGISTER_OP_CPU_KERNEL(c_allreduce_sum,
                       ops::CAllReduceOpCPUKernel<ops::kRedSum, float>,
                       ops::CAllReduceOpCPUKernel<ops::kRedSum, double>,
                       ops::CAllReduceOpCPUKernel<ops::kRedSum, int>,
                       ops::CAllReduceOpCPUKernel<ops::kRedSum, int64_t>,
                       ops::CAllReduceOpCPUKernel<ops::kRedSum, plat::float16>)
