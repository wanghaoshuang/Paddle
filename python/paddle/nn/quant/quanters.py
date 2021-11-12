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

import paddle.nn as nn
from paddle.fluid.dygraph.layers import Layer
from paddle.fluid import core
from paddle.fluid import dygraph_utils
from paddle.fluid import unique_name
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.framework import _varbase_creator
from paddle.fluid.framework import in_dygraph_mode
from paddle.fluid.initializer import Constant
from paddle.fluid.data_feeder import check_variable_and_dtype
from paddle.nn import functional as F
import logging
from paddle.fluid.log_helper import get_logger
from paddle import _C_ops

__all__ = [
    'get_fake_quant_type',
    'FakeQuantAbsMax',
    'FakeQuantMovingAverageAbsMax',
    'FakeQuantChannelWiseAbsMax',
    'MovingAverageAbsMaxScale',
    'MAOutputScaleLayer',
    'FakeQuantMAOutputScaleLayer',
    'QuantStub',
]

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')


class FakeQuantAbsMax(Layer):
    r"""
    FakeQuantAbsMax layer does the abs_max quant and then dequant.
    Its computational formula is described as below:

    :math:`scale = max(abs(X))`
    :math:`range = 2^{bit\_length - 1} - 1`
    :math:`Out = round(X / scale * range) * scale / range`
    """

    def __init__(self,
                 name=None,
                 quant_bits=8,
                 dtype='float32',
                 quant_on_weight=False,
                 onnx_style=False):
        super(FakeQuantAbsMax, self).__init__()
        self._onnx_style = onnx_style
        self._quant_bits = quant_bits
        self._name = name
        scale_prefix = "{}.scale".format(
            name) if name else 'quant_dequant.scale'
        self._scale_name = unique_name.generate(scale_prefix)
        if quant_on_weight:
            scale_attr = ParamAttr(
                name=self._scale_name,
                initializer=Constant(0.0),
                trainable=False)
            self._scale = self.create_parameter(
                shape=[1], attr=scale_attr, dtype=self._dtype)

            self.add_parameter("_scale", self._scale)
            self._scale.stop_gradient = True
        else:
            self._scale = None

    def forward(self, input):
        if in_dygraph_mode():
            attrs = ('bit_length', self._quant_bits)
            quant_out = _varbase_creator(
                type=input.type,
                name="{}.quantized.dequantized".format(input.name),
                shape=input.shape,
                dtype=input.dtype,
                persistable=False)
            out_scale = self._scale
            if not out_scale:
                out_scale = _varbase_creator(
                    type=core.VarDesc.VarType.LOD_TENSOR,
                    name=self._scale_name,
                    shape=[1],
                    dtype=self._dtype,
                    persistable=False)
                out_scale.stop_gradient = True
            out, _, = _C_ops.fake_quantize_dequantize_abs_max(input, quant_out,
                                                              out_scale, *attrs)
            return out

        check_variable_and_dtype(input, 'input', ['float32'], "FakeQuantAbsMax")
        attrs = {'bit_length': self._quant_bits}
        inputs = {"X": [input]}
        quant_out = self._helper.create_variable(
            name="{}.quantized.dequantized".format(input.name),
            dtype=input.dtype,
            type=core.VarDesc.VarType.LOD_TENSOR,
            persistable=False,
            stop_gradient=False)
        out_scale = self._scale
        if not out_scale:
            out_scale = self._helper.create_variable(
                name=self._scale_name,
                dtype=self._dtype,
                type=core.VarDesc.VarType.LOD_TENSOR,
                persistable=False,
                stop_gradient=True)
        outputs = {"Out": [quant_out], "OutScale": [out_scale]}
        assert not self._onnx_style, "Unsorpported onnx style for abs_max!" 
        self._helper.append_op(
            type="fake_quantize_dequantize_abs_max",
            inputs=inputs,
            outputs=outputs,
            attrs=attrs)

        return quant_out


class FakeQuantMovingAverageAbsMax(nn.Layer):
    r"""
    FakeQuantMovingAverageAbsMax layer does the moving_average_abs_max quant and then dequant.
    Its computational formula is described as below:

    :math:`scale = (moving\_rate*accum+max(abs(x)))/(moving\_rate*state+1)`
    :math:`range = 2^{bit\_length - 1} - 1`
    :math:`Out = round(X / scale * range) * scale / range`
    """

    def __init__(self,
                 name=None,
                 moving_rate=0.9,
                 quant_bits=8,
                 dtype='float32',
                 onnx_style=False):
        super(FakeQuantMovingAverageAbsMax, self).__init__()
        self._moving_rate = moving_rate
        self._quant_bits = quant_bits
        self._onnx_style = onnx_style

        scale_prefix = "{}.scale".format(
            name) if name else 'quant_dequant.scale'
        scale_attr = ParamAttr(
            name=unique_name.generate(scale_prefix),
            initializer=Constant(0.001),
            trainable=False)
        self._scale = self.create_parameter(
            shape=[1], attr=scale_attr, dtype=dtype)
        self.add_parameter("_scale", self._scale)
        self._scale.stop_gradient = True

        state_prefix = "{}.state".format(
            name) if name else 'quant_dequant.state'
        state_attr = ParamAttr(
            name=unique_name.generate(state_prefix),
            initializer=Constant(1),
            trainable=False)
        self._state = self.create_parameter(
            shape=[1], attr=state_attr, dtype=dtype)
        self._state.stop_gradient = True

        accum_prefix = "{}.accum".format(
            name) if name else 'quant_dequant.accum'
        accum_attr = ParamAttr(
            name=unique_name.generate(accum_prefix),
            initializer=Constant(1),
            trainable=False)
        self._accum = self.create_parameter(
            shape=[1], attr=accum_attr, dtype=dtype)
        self._accum.stop_gradient = True

    def forward(self, input):
        if in_dygraph_mode():
            attrs = ('moving_rate', self._moving_rate, 'bit_length',
                     self._quant_bits, 'is_test', not self.training)
            quant_out = _varbase_creator(
                type=input.type,
                name="{}.quantized.dequantized".format(input.name),
                shape=input.shape,
                dtype=input.dtype,
                persistable=False)
            state = self._state if self.training else None
            accum = self._accum if self.training else None

            out, _, _, _ = _C_ops.fake_quantize_dequantize_moving_average_abs_max(
                input, self._scale, accum, state, quant_out, self._scale, state,
                accum, *attrs)
            return out

        check_variable_and_dtype(input, 'input', ['float32'],
                                 "FakeQuantMovingAverageAbsMax")
        attrs = {
            'moving_rate': self._moving_rate,
            'bit_length': self._quant_bits,
            'is_test': not self.training
        }
        inputs = {"X": [input], "InScale": [self._scale]}

        quant_out = self._helper.create_variable(
            name="{}.quantized".format(input.name),
            dtype=input.dtype,
            type=core.VarDesc.VarType.LOD_TENSOR,
            persistable=False,
            stop_gradient=False)

        outputs = {"Out": [quant_out], "OutScale": [self._scale]}
        if self.training:
            inputs['InState'] = [self._state]
            inputs['InAccum'] = [self._accum]
            outputs['OutState'] = [self._state]
            outputs['OutAccum'] = [self._accum]
        
        if self._onnx_style:
            dequant_out = self._helper.create_variable(
                name="{}.quantized.dequantized".format(input.name),
                dtype=input.dtype,
                type=core.VarDesc.VarType.LOD_TENSOR,
                persistable=False,
                stop_gradient=False)
            self._helper.append_op(
                type="fake_quantize_moving_average_abs_max",
                inputs=inputs,
                outputs=outputs,
                attrs=attrs)
            self._helper.append_op(
                type="fake_dequantize_max_abs",
                inputs={"X": quant_out, "Scale": self._scale},
                outputs={"Out": dequant_out},
                attrs={"max_range": 127})
            return dequant_out
        else:

            self._helper.append_op(
                type="fake_quantize_moving_average_abs_max",
                inputs=inputs,
                outputs=outputs,
                attrs=attrs)
            return quant_out




class FakeQuantChannelWiseAbsMax(Layer):
    def __init__(self,
                 name=None,
                 channel_num=None,
                 quant_bits=8,
                 quant_axis=0,
                 dtype='float32',
                 quant_on_weight=False,
                 onnx_style=False):
        assert quant_on_weight == True, "Channel_wise only can be used on weight quantization."
        super(FakeQuantChannelWiseAbsMax, self).__init__()
        self._onnx_style = onnx_style
        self._quant_bits = quant_bits
        self._quant_axis = quant_axis
        self._dtype = dtype
        self._name = name
        self._channel_num = channel_num
        scale_prefix = "{}.scale".format(
            name) if name else 'quant_dequant.scale'
        self._scale_name = unique_name.generate(scale_prefix)
        if quant_on_weight:
            scale_attr = ParamAttr(
                name=self._scale_name,
                initializer=Constant(0.0),
                trainable=False)
            self._scale = self.create_parameter(
                shape=[self._channel_num], attr=scale_attr, dtype=self._dtype)
            self.add_parameter("_scale", self._scale)
            self._scale.stop_gradient = True
        else:
            self._scale = None

    def forward(self, input):
        if in_dygraph_mode():
            attrs = ('bit_length', self._quant_bits, 'quant_axis',
                     self._quant_axis)
            quant_out = _varbase_creator(
                type=input.type,
                name="{}.quantized.dequantized".format(input.name),
                shape=input.shape,
                dtype=input.dtype,
                persistable=False)

            out_scale = self._scale
            if out_scale is None:
                out_scale = _varbase_creator(
                    type=core.VarDesc.VarType.LOD_TENSOR,
                    name=self._scale_name,
                    shape=[self._channel_num],
                    dtype=self._dtype,
                    persistable=False)
                out_scale.stop_gradient = True

            out, _, = _C_ops.fake_channel_wise_quantize_dequantize_abs_max(
                input, quant_out, out_scale, *attrs)
            return out

        check_variable_and_dtype(input, 'input', ['float32'],
                                 "FakeQuantChannelWiseAbsMax")
        attrs = {'bit_length': self._quant_bits, 'quant_axis': self._quant_axis}
        inputs = {"X": [input]}


        quant_out = self._helper.create_variable(
            name="{}.quantized".format(input.name),
            dtype=input.dtype,
            type=core.VarDesc.VarType.LOD_TENSOR,
            persistable=False,
            stop_gradient=False)
        out_scale = self._scale
        if not out_scale:
            out_scale = self._helper.create_variable(
                name=self._scale_name,
                dtype=self._dtype,
                type=core.VarDesc.VarType.LOD_TENSOR,
                persistable=False,
                stop_gradient=True)
        outputs = {"Out": [quant_out], "OutScale": [out_scale]}
        if self._onnx_style:
            dequant_out = self._helper.create_variable(
                name="{}.quantized.dequantized".format(input.name),
                dtype=input.dtype,
                type=core.VarDesc.VarType.LOD_TENSOR,
                persistable=False,
                stop_gradient=False)

            self._helper.append_op(
                type="fake_channel_wise_quantize_abs_max",
                inputs=inputs,
                outputs={"Out": [quant_out], "OutScale": [out_scale]},
                attrs=attrs)
    
            self._helper.append_op(
                type="fake_channel_wise_dequantize_max_abs",
                inputs={"X": [quant_out], "Scales": [self._scale]},
                outputs={"Out": [dequant_out]},
                attrs={"quant_bits": [self._quant_bits], 
                       "quant_axis": self._quant_axis})
            return dequant_out

        else:
            self._helper.append_op(
                type="fake_channel_wise_quantize_dequantize_abs_max",
                inputs=inputs,
                outputs=outputs,
                attrs=attrs)
            return quant_out



class MovingAverageAbsMaxScale(nn.Layer):
    def __init__(self, name=None, moving_rate=0.9, dtype='float32'):
        r"""
        MovingAverageMaxScale layer is used to calculating the output quantization
        scale of Layer. Its computational formula is described as below:

        :math:`scale = (moving\_rate*accum+max(abs(x)))/(moving\_rate*state+1)`
        :math:`Out = X`
        """
        super(MovingAverageAbsMaxScale, self).__init__()
        self._moving_rate = moving_rate

        scale_prefix = '{}.scale'.format(name) if name else 'outscale.scale'
        scale_name = unique_name.generate(scale_prefix)
        scale_attr = ParamAttr(
            name=scale_name, initializer=Constant(0), trainable=False)
        self._scale = self.create_parameter(
            shape=[1], attr=scale_attr, dtype=dtype)
        self._scale.stop_gradient = True

        state_prefix = "{}.state".format(name) if name else 'outscale.state'
        state_attr = ParamAttr(
            name=unique_name.generate(state_prefix),
            initializer=Constant(0),
            trainable=False)
        self._state = self.create_parameter(
            shape=[1], attr=state_attr, dtype=dtype)
        self._state.stop_gradient = True

        accum_prefix = "{}.accum".format(name) if name else 'outscale.accum'
        accum_attr = ParamAttr(
            name=unique_name.generate(accum_prefix),
            initializer=Constant(0),
            trainable=False)
        self._accum = self.create_parameter(
            shape=[1], attr=accum_attr, dtype=dtype)
        self._accum.stop_gradient = True

    def forward(self, input):
        if in_dygraph_mode():
            attrs = ('moving_rate', self._moving_rate, 'is_test',
                     not self.training)
            state = self._state if self.training else None
            accum = self._accum if self.training else None
            quant_out = _varbase_creator(
                type=input.type,
                name="{}.tmp".format(input.name),
                shape=input.shape,
                dtype=input.dtype,
                persistable=False)

            out, _, _, _ = _C_ops.moving_average_abs_max_scale(
                input, accum, state, quant_out, self._scale, state, accum,
                *attrs)
            return out

        check_variable_and_dtype(input, 'input', ['float32', 'float64'],
                                 'MovingAverageAbsMaxScale')

        attrs = {'moving_rate': self._moving_rate, 'is_test': not self.training}
        inputs = {"X": [input]}
        quant_out = self._helper.create_variable(
            name="{}.tmp".format(input.name),
            dtype=input.dtype,
            type=core.VarDesc.VarType.LOD_TENSOR,
            persistable=False,
            stop_gradient=False)
        outputs = {"Out": [quant_out], "OutScale": [self._scale]}

        if self.training:
            inputs['InState'] = [self._state]
            inputs['InAccum'] = [self._accum]
            outputs['OutState'] = [self._state]
            outputs['OutAccum'] = [self._accum]

        self._helper.append_op(
            type="moving_average_abs_max_scale",
            inputs=inputs,
            outputs=outputs,
            attrs=attrs)

        return quant_out


QuantStub = MovingAverageAbsMaxScale


class MAOutputScaleLayer(nn.Layer):
    """
    Add MovingAverageMaxScale layer to the behind of the input layer.
    Calculate the scale (moving average abs max) for the output of the input layer.
    """

    def __init__(self, layer=None, moving_rate=0.9, name=None, dtype='float32'):
        r"""
        Construct
        """
        super(MAOutputScaleLayer, self).__init__()
        self._layer = layer
        if name is None:
            name = layer.full_name()
        self._ma_output_scale = \
            MovingAverageAbsMaxScale(name, moving_rate, dtype)

    def forward(self, *inputs, **kwargs):
        out = self._layer(*inputs, **kwargs)
        # TODO (jc): support the ops of several outputs
        if (isinstance(out, list) or isinstance(out, tuple) or
                isinstance(out, dict)):
            return out
        else:
            return self._ma_output_scale(out)


class FakeQuantMAOutputScaleLayer(nn.Layer):
    """
    Add FakeQuantMovingAverageAbsMax layer to the behind of the input layer.
    """

    def __init__(self,
                 layer,
                 weight_bits=8,
                 activation_bits=8,
                 moving_rate=0.9,
                 name=None,
                 *args,
                 **kwargs):

        super(FakeQuantMAOutputScaleLayer, self).__init__()
        self._layer = layer
        self._fake_quant_output = _get_fake_quant_type(
            'moving_average_abs_max',
            name=layer.full_name() if name is None else name,
            moving_rate=moving_rate,
            quant_bits=activation_bits,
            dtype=self._dtype,
            quant_on_weight=False)

    def forward(self, *inputs, **kwargs):
        out = self._layer(*inputs, **kwargs)
        # TODO (jc): support the ops of several outputs
        if (isinstance(out, list) or isinstance(out, tuple)) and len(out) > 1:
            return out
        else:
            return self._fake_quant_output(out)


def get_fake_quant_type(quant_type, **kwargs):
    call_args = {
        "name": kwargs.get("name", None),
        "quant_bits": kwargs.get("quant_bits", 8),
        "dtype": kwargs.get("dtype", "float32"),
        "onnx_style": kwargs.get("onnx_style", False)
    }

    if quant_type == 'abs_max':
        call_args["quant_on_weight"] = kwargs.get("quant_on_weight", False)
    elif quant_type == 'moving_average_abs_max':
        call_args["moving_rate"] = kwargs.get("moving_rate", 0.9)
    elif quant_type == 'channel_wise_abs_max':
        call_args["quant_on_weight"] = kwargs.get("quant_on_weight", False)
        call_args["channel_num"] = kwargs.get("channel_num", None)
        call_args["quant_axis"] = kwargs.get("quant_axis", 0)
        assert call_args["channel_num"] is not None, (
            "You need to input channel_num"
            "when you use channel_wise_abs_max strategy.")
    fake_quant_map = {
        'abs_max': FakeQuantAbsMax,
        'moving_average_abs_max': FakeQuantMovingAverageAbsMax,
        'channel_wise_abs_max': FakeQuantChannelWiseAbsMax
    }

    return fake_quant_map[quant_type](**call_args)
