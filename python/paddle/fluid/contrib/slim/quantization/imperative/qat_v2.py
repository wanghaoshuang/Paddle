#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import logging
import numpy as np
import sys
import os
import warnings

import paddle
import paddle.nn.quant.quant_layers as quant_layers
from paddle.nn.quant.quant_wrapper import QuantizedWrapper
from paddle.fluid import dygraph, core, framework, unique_name
from paddle.fluid.executor import Executor, global_scope
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Constant
from paddle.fluid.dygraph.io import INFER_MODEL_SUFFIX, INFER_PARAMS_SUFFIX
from paddle.fluid.io import load_inference_model, save_inference_model
from paddle.fluid.framework import IrGraph
from paddle.fluid.log_helper import get_logger
from ..quantization_pass import ReplaceFakeQuantDequantPass, QuantWeightPass
from . import utils

__all__ = ['ImperativeQuantAwareV2']

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')


class ImperativeQuantAwareV2(object):
    """
    Applying quantization aware training (QAT) to the dgraph model.
    """

    def __init__(
            self,
            quantizable_layer_type=['Conv2D', 'Linear', 'Conv2DTranspose'],
            weight_quantize_type='abs_max',
            activation_quantize_type='moving_average_abs_max',
            weight_bits=8,
            activation_bits=8,
            moving_rate=0.9,
            weight_preprocess_layer=None,
            act_preprocess_layer=None,
            weight_quantize_layer=None,
            act_quantize_layer=None):
        """
        The constructor for ImperativeQuantAware.

        Args:
            quantizable_layer_type(list[str | layer]): List the type of
                layers that will be quantized. Default is ['Conv2D', 'Linear'].
            weight_quantize_type(str): quantization type for weights,
                which supports 'abs_max' and 'channel_wise_abs_max'.
            activation_quantize_type(str): quantization type for activations,
                which supports 'abs_max' and 'moving_average_abs_max' now.
                If using 'abs_max' mode, the quantization scale will be
                calculated dynamically each step in both training and testing
                period. If using 'moving_average_abs_max', the static
                quantization scale will be calculated during training and
                used in inference.
            weight_bits(int): quantization bit number for weights, whereas
                the bias is not quantized.
            activation_bits(int): quantization bit number for activations.
            moving_rate(float): the parameter for 'moving_average_abs_max'
                quantization.
            weight_preprocess_layer(paddle.nn.Layer, optional): A paddle
                Layer that defines how to preprocess weight before quantization.
                Using this can quickly test if user's preprocess method works
                or not. The input is non-quantized weight and function returns
                processed weight to be quantized.
                If None, the weight will be quantized directly.
                Default is None.
            act_preprocess_layer(paddle.nn.Layer, optional): A paddle Layer
                that defines how to preprocess activation before quantization.
                Using this can quickly test if user's preprocess method works
                or not. The input is non-quantized activation and function returns
                processed activation to be quantized.
                If None, the activation will be quantized directly.
                Default is None.
            weight_quantize_layer(paddle.nn.Layer, optional): A paddle Layer that
                defines how to quantize weight.
                Using this can quickly test if user's quantization method works or not.
                In this layer, user should both define quantization method and
                dequantization method, that is, the function's input is non-quantized
                weight and returns dequantized weight.
                If None, will use uantization op defined by 'weight_quantize_type'.
                Default is None.
            act_quantize_layer(paddle.nn.Layer, optional): A paddle Layer that defines
                how to quantize activation.
                Using this can quickly test if user's quantization method works or not.
                In this layer, user should both define quantization method and
                dequantization method, that is, the function's input is non-quantized
                activation and returns dequantized activation. 
                If None, will use quantization op defined by 'activation_quantize_type'.
                Default is None.

        Note:
            If user sets attribute 'skip_quant' to a Layer that support dynamic
            quantization and sets it to true, the layer would not be quantized
            during training. If this attribute is not sets or the attribute is
            false, the Layer would be qunatized in training.

        Examples 1:
        .. code-block:: python

            import paddle
            from paddle.fluid.contrib.slim.quantization \
                import ImperativeQuantAware
            from paddle.vision.models \
                import resnet
            
            model = resnet.resnet50(pretrained=True)

            imperative_qat = ImperativeQuantAware(
                weight_quantize_type='abs_max',
                activation_quantize_type='moving_average_abs_max')
            
            # Add the fake quant logical.
            # The original model will be rewrite.
            # The outscale of outputs in supportted layers would be calculated.
            imperative_qat.quantize(model)

            # Fine-tune the quantized model
            # ...
            
            # Save quant model for the inference.
            imperative_qat.save_quantized_model(
                layer=model,
                model_path="./resnet50_qat",
                input_spec=[
                    paddle.static.InputSpec(
                    shape=[None, 3, 224, 224], dtype='float32')])

        Examples 2:
        .. code-block:: python

            import paddle
            from paddle.fluid.contrib.slim.quantization \
                import ImperativeQuantAware

            class ImperativeModel(paddle.nn.Layer):
                def __init__(self):
                    super(ImperativeModel, self).__init__()
                    # self.linear_0 would skip the quantization.
                    self.linear_0 = paddle.nn.Linear(784, 400)
                    self.linear_0.skip_quant = True

                    # self.linear_1 would not skip the quantization.
                    self.linear_1 = paddle.nn.Linear(400, 10)
                    self.linear_1.skip_quant = False

                def forward(self, inputs):
                    x = self.linear_0(inputs)
                    x = self.linear_1(inputs)
                    return x

            model = ImperativeModel()
            imperative_qat = ImperativeQuantAware(
                weight_quantize_type='abs_max',
                activation_quantize_type='moving_average_abs_max')

            # Add the fake quant logical.
            # The original model will be rewrite.
            #
            # There is only one Layer(self.linear1) would be added the
            # fake quant logical.
            imperative_qat.quantize(model)

            # Fine-tune the quantized model
            # ...

            # Save quant model for the inference.
            imperative_qat.save_quantized_model(
                layer=model,
                model_path="./imperative_model_qat")
        """
        super(ImperativeQuantAwareV2, self).__init__()

        kwargs = {
            "quantizable_layer_type": quantizable_layer_type,
            "weight_quantize_type": weight_quantize_type,
            "activation_quantize_type": activation_quantize_type,
            "weight_bits": weight_bits,
            "activation_bits": activation_bits,
            "moving_rate": moving_rate,
            "weight_preprocess_layer": weight_preprocess_layer,
            "act_preprocess_layer": act_preprocess_layer,
            "weight_quantize_layer": weight_quantize_layer,
            "act_quantize_layer": act_quantize_layer
        }

        self._quantize_inputs = ImperativeQuantizeInputs(**kwargs)

#        self._quantize_outputs = ImperativeQuantizeOutputs(moving_rate)

    def quantize(self, model):
        """
        According to weights' and activations' quantization types,
        the model will be added some fake quant ops, such as
        fake_quantize_dequantize_moving_average_abs_max,
        fake_quantize_dequantize_abs_max and so on. At the same time,
        the out_scale value of outputs would be calculated.

        Args:
            model(paddle.nn.Layer): the model to be quantized.
        Returns:
            None

        Examples:
        .. code-block:: python

            import paddle
            from paddle.fluid.contrib.slim.quantization \
                import ImperativeQuantAware

            class ImperativeModel(paddle.nn.Layer):
                def __init__(self):
                    super(ImperativeModel, self).__init__()
                    # self.linear_0 would skip the quantization.
                    self.linear_0 = paddle.nn.Linear(784, 400)
                    self.linear_0.skip_quant = True

                    # self.linear_1 would not skip the quantization.
                    self.linear_1 = paddle.nn.Linear(400, 10)
                    self.linear_1.skip_quant = False

                def forward(self, inputs):
                    x = self.linear_0(inputs)
                    x = self.linear_1(inputs)
                    return x

            model = ImperativeModel()
            imperative_qat = ImperativeQuantAware(
                weight_quantize_type='abs_max',
                activation_quantize_type='moving_average_abs_max')

            # Add the fake quant logical.
            # The original model will be rewrite.
            #
            # There is only one Layer(self.linear1) would be added the
            # fake quant logical.
            imperative_qat.quantize(model)
        """
        assert isinstance(model, dygraph.Layer), \
            "The model must be the instance of dygraph.Layer."


        self._parameters = set([param.name for param in model.parameters()])
        print(self._parameters)
        self._quantize_inputs.apply(model)

    def _quant(self, x, scale, num_bits, quant_axis):
        assert quant_axis in [0, 1], 'quant_axis should be 0 or 1 for now.'
        bnt = (1 << (num_bits - 1)) - 1

        def _clip(x, scale):
            x[x > scale] = scale
            x[x < -scale] = -scale
            return x

        if isinstance(scale, list):
            for i, s in enumerate(scale):
                if s == 0.0:
                    s = 1e-8
                if quant_axis == 0:
                    x[i] = _clip(x[i], s)
                    x[i] = np.round(x[i] / s * bnt)
                else:
                    x[:, i] = _clip(x[:, i], s)
                    x[:, i] = np.round(x[:, i] / s * bnt)
        else:
            scale = 1e-8 if scale == 0.0 else scale
            x = _clip(x, scale)
            x = np.round(x / scale * bnt)
        return x


    def _quant_weights(self, program, scope):

        fake_quant_ops = ["fake_channel_wise_quantize_abs_max"]
        param_names = set()
        
        target_ops = []
        for block in program.blocks:
            del_idx = []
            for i, op in enumerate(block.ops):
                if op.type in fake_quant_ops:
                    in_name = op.input("X")[0]
                    if in_name in self._parameters:
                        num_bits = op.attr("bit_length")
                        quant_axis = op.attr("quant_axis")
                        out_name = op.output("Out")[0]
                        scale_name = op.output("OutScale")[0]
                        scale_v = utils.load_variable_data(scope, scale_name)
                        scale_v = utils.fp_numpy_to_naive(scale_v)
                        weight_t = scope.find_var(in_name).get_tensor()
                        weight_v = np.array(weight_t)
        #                print(f"weight_v: {weight_v.shape}; scale_v: {scale_v.sha}; num_bits: {num_bits}; quant_axis: {quant_axis}")
                        quanted_weight = self._quant(weight_v, scale_v, num_bits, quant_axis)
                        weight_t.set(quanted_weight, self._place)
                        
                        next_ops = utils.find_next_ops(op.block, out_name)
                        for _op in next_ops:
                            _op._rename_input(out_name, in_name)
                        op._rename_input(in_name, "")
                        op._rename_output(scale_name, "")
                        op._rename_output(out_name, "")
#                        del_idx.append(i)
#            block.ops = [_op for i, _op in enumerate(block.ops) if i not in del_idx] 
                
    def _set_skip_quant_attr(self, program):
        """
        Label the skip quantized ops.
        """
        for block in program.blocks:
            for op in block.ops:
                if self._is_skip_quant_op(block, op):
                    op._set_attr("skip_quant", True)
                    op._set_attr("with_quant_attr", True)

    def _is_skip_quant_op(self, block, in_op):
        """
        The input op should be skipped quantization.
        1. the type of input op should be conv2d, depthwise_conv2d or matmul
        2. the previous ops of the input op are not fake_quantize_dequantize ops
        """
        target_op_types = [
            "conv2d", "depthwise_conv2d", "matmul", "conv2d_transpose"
        ]
        if in_op.type not in target_op_types:
            return False

        previous_ops = [utils.find_previous_op(block, arg_name) \
            for arg_name in in_op.input_arg_names]
        return any(op is not None and op.type not in \
            utils.fake_quantize_dequantize_op_types for op in previous_ops)


    def save_quantized_model(self, layer, path, input_spec=None, **config):
        """
        Save the quantized model for the inference.

        Args:
            model (Layer): The model to be saved.
            path (str): The path prefix to save model. The format is 
                ``dirname/file_prefix`` or ``file_prefix``.
            input_spec (list[InputSpec|Tensor], optional): Describes the input
                of the saved model's forward method, which can be described by
                InputSpec or example Tensor. If None, all input variables of 
                the original Layer's forward method would be the inputs of
                the saved model. Default None.
            **configs (dict, optional): Other save configuration options for
                compatibility. We do not recommend using these configurations,
                they may be removed in the future. If not necessary, DO NOT use
                them. Default None.
                The following options are currently supported:
                (1) output_spec (list[Tensor]): Selects the output targets of
                the saved model. By default, all return variables of original
                Layer's forward method are kept as the output of the saved model.
                If the provided ``output_spec`` list is not all output variables, 
                the saved model will be pruned according to the given
                ``output_spec`` list. 

        Returns:
            None
        """
        assert isinstance(layer, dygraph.Layer), \
            "The model must be the instance of dygraph.Layer."
        paddle.jit.save(layer=layer, path=path, input_spec=input_spec, **config)

        is_dynamic_mode = False
        if paddle.in_dynamic_mode():
            is_dynamic_mode = True
            paddle.enable_static()

        self._place = core.CPUPlace()
        scope = global_scope()
        exe = Executor(self._place)

        dirname = os.path.dirname(path)
        basename = os.path.basename(path)
        model_filename = basename + INFER_MODEL_SUFFIX
        params_filename = basename + INFER_PARAMS_SUFFIX

        [infer_program, feed_target_names, fetch_targets] = (
            load_inference_model(
                dirname=dirname,
                executor=exe,
                model_filename=model_filename,
                params_filename=params_filename))

        #self._quant_weights(infer_program, scope)

#        self._set_skip_quant_attr(infer_program)

        graph = IrGraph(core.Graph(infer_program.desc), for_test=False)
        transform_pass = ReplaceFakeQuantDequantPass(scope, self._place)
        transform_pass.apply(graph)

        quant_weight_pass = QuantWeightPass(scope, self._place)
        quant_weight_pass.apply(graph)
        infer_program = graph.to_program()

        save_inference_model(
            dirname=dirname,
            feeded_var_names=feed_target_names,
            target_vars=fetch_targets,
            executor=exe,
            main_program=infer_program.clone(),
            model_filename=model_filename,
            params_filename=params_filename,
            clip_extra=True)

        if is_dynamic_mode:
            paddle.disable_static()



class ImperativeQuantizeInputs(object):
    """
    Based on the input params, add the quant_dequant computational
    logic both for activation inputs and weight inputs.
    """

    def __init__(
            self,
            quantizable_layer_type=['Conv2D', 'Linear', 'Conv2DTranspose'],
            weight_quantize_type='abs_max',
            activation_quantize_type='moving_average_abs_max',
            weight_bits=8,
            activation_bits=8,
            moving_rate=0.9,
            weight_preprocess_layer=None,
            act_preprocess_layer=None,
            weight_quantize_layer=None,
            act_quantize_layer=None,):
        """
        The constructor for ImperativeQuantizeInputs. 

        Please refer to the args of ImperativeQuantAware.
        """
        super(ImperativeQuantizeInputs, self).__init__()

        self._quantizable_layer_type = tuple(
            utils.layer_name_map[layer]
            if layer in utils.layer_name_map else layer
            for layer in quantizable_layer_type)
        for layer in self._quantizable_layer_type:
            assert not isinstance(layer, str) \
                and layer in utils.fake_quant_input_layers, \
                "%s is unspported to be quantized." % layer

        quantize_type = {
            'abs_max', 'moving_average_abs_max', 'channel_wise_abs_max'
        }
        assert weight_quantize_type != 'moving_average_abs_max' \
            and weight_quantize_type in quantize_type, \
            "Unsupported weight_quantize_type: %s. It can only " \
            "be abs_max or channel_wise_abs_max." % weight_quantize_type
        # TODO (jc): activation_quantize_type supports range_abs_max
        assert activation_quantize_type == 'moving_average_abs_max', \
            "Unsupported activation_quantize_type: %s. It can " \
            "only be moving_average_abs_max now." \
            % activation_quantize_type

        bits_check = lambda bits: isinstance(bits, int) \
            and bits >= 0 and bits <= 16
        assert bits_check(weight_bits), \
            "weight_bits should be 1, 2,... or 16."
        assert bits_check(activation_bits), \
            "activation_bits should be 1, 2,... or 16."

        layer_check = lambda method: method is None or \
            issubclass(method, dygraph.layers.Layer)
        assert layer_check(weight_preprocess_layer), \
            "weight_preprocess should be nn.Layer."
        assert layer_check(act_preprocess_layer), \
            "act_preprocess should be nn.Layer."
        assert layer_check(weight_quantize_layer), \
            "weight_quantize should be nn.Layer."
        assert layer_check(act_quantize_layer), \
            "act_quantize should be nn.Layer."

        self._kwargs = {
            "weight_quantize_type": weight_quantize_type,
            "activation_quantize_type": activation_quantize_type,
            "weight_bits": weight_bits,
            "activation_bits": activation_bits,
            "moving_rate": moving_rate,
            "weight_pre_layer": weight_preprocess_layer,
            "act_pre_layer": act_preprocess_layer,
#            "weight_quant_layer": weight_quantize_layer,
#            "act_quant_layer": act_quantize_layer,
            "onnx_style": True,
        }

    def apply(self, model):
        """
        Quantize the weights and activations to calculate for specific 
        layers.

        Args:
            model(paddle.nn.Layer): The target model which would
                calculate the input quantization scale.

        Returns:
            None
        """

        assert isinstance(model, dygraph.Layer), \
            "The model must be the instance of dygraph.Layer."

        for name, cur_layer in model.named_sublayers():
            if not isinstance(cur_layer, self._quantizable_layer_type) \
                or (hasattr(cur_layer, "skip_quant") \
                    and cur_layer.skip_quant == True):
                continue

            parent_layer, sub_name = \
                utils.find_parent_layer_and_sub_name(model, name)

            cur_quant_layer = self._get_input_quantized_layer(cur_layer)
            setattr(parent_layer, sub_name, cur_quant_layer)

    def _get_input_quantized_layer(self, layer):
        quant_layer_name = None

        for key, value in utils.layer_name_map.items():
            if isinstance(layer, value):
                quant_layer_name = 'Quantized' + key
                break
        assert quant_layer_name is not None, \
            "The layer %s is unsupported to be quantized." \
            % layer.full_name()

        return QuantizedWrapper(layer, **self._kwargs)



