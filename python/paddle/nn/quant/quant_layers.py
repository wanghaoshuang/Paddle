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

from paddle.fluid.dygraph import layers
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

from .quanters import get_fake_quant_type

__all__ = [
    'QuantizedConv2D',
    'QuantizedConv2DTranspose',
    'QuantizedLinear',
]

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')


class QuantizedConv2D(layers.Layer):
    """
    The computational logic of QuantizedConv2D is the same with Conv2D.
    The only difference is that its inputs are all fake quantized.
    """

    def __init__(self,
                 layer,
                 weight_bits=8,
                 activation_bits=8,
                 moving_rate=0.9,
                 weight_quantize_type='abs_max',
                 activation_quantize_type='abs_max',
                 weight_pre_layer=None,
                 act_pre_layer=None,
                 weight_quant_layer=None,
                 act_quant_layer=None,
                 onnx_style=False,):
        super(QuantizedConv2D, self).__init__()
        # For Conv2D
        self._groups = getattr(layer, '_groups')
        self._stride = getattr(layer, '_stride')
        self._padding = getattr(layer, '_padding')
        self._padding_mode = getattr(layer, '_padding_mode')
        if self._padding_mode != 'zeros':
            self._reversed_padding_repeated_twice = getattr(
                layer, '_reversed_padding_repeated_twice')
        self._dilation = getattr(layer, '_dilation')
        self._data_format = getattr(layer, '_data_format')
        self.weight = getattr(layer, 'weight')
        self.bias = getattr(layer, 'bias')

        # For FakeQuant
        self._conv2d_quant_axis = 0
        if weight_quant_layer is not None:
            self._fake_quant_weight = weight_quant_layer()
        else:
            self._fake_quant_weight = get_fake_quant_type(
                weight_quantize_type,
                name=self.weight.name,
                moving_rate=moving_rate,
                quant_bits=weight_bits,
                dtype=self._dtype,
                quant_on_weight=True,
                channel_num=self.weight.shape[self._conv2d_quant_axis],
                quant_axis=self._conv2d_quant_axis)
        if act_quant_layer is not None:
            self._fake_quant_input = act_quant_layer()
        else:
            self._fake_quant_input = get_fake_quant_type(
                activation_quantize_type,
                name=layer.full_name(),
                moving_rate=moving_rate,
                quant_bits=activation_bits,
                dtype=self._dtype,
                quant_on_weight=False)

        self._act_preprocess = act_pre_layer(
        ) if act_pre_layer is not None else None
        self._weight_preprocess = weight_pre_layer(
        ) if weight_pre_layer is not None else None

    def forward(self, input):
        if self._act_preprocess is not None:
            input = self._act_preprocess(input)
        quant_input = self._fake_quant_input(input)

        weight = self.weight
        if self._weight_preprocess is not None:
            weight = self._weight_preprocess(self.weight)
        quant_weight = self._fake_quant_weight(weight)

        if self._padding_mode != 'zeros':
            quant_input = F.pad(quant_input,
                                self._reversed_padding_repeated_twice,
                                mode=self._padding_mode,
                                data_format=self._data_format)
            self._padding = 0

        return F.conv2d(
            quant_input,
            quant_weight,
            bias=self.bias,
            padding=self._padding,
            stride=self._stride,
            dilation=self._dilation,
            groups=self._groups,
            data_format=self._data_format)


class QuantizedConv2DTranspose(layers.Layer):
    """
    The computational logic of QuantizedConv2DTranspose is the same with Conv2DTranspose.
    The only difference is that its inputs are all fake quantized.
    
    Examples:
       .. code-block:: python
          import paddle
          import paddle.nn as nn
          from paddle.nn.quant.quant_layers import QuantizedConv2DTranspose
          x_var = paddle.uniform((2, 4, 8, 8), dtype='float32', min=-1., max=1.)
          conv = nn.Conv2DTranspose(4, 6, (3, 3))
          conv_quantized = QuantizedConv2DTranspose(conv)
          y_quantized = conv_quantized(x_var)
          y_var = conv(x_var)
          y_quantized_np = y_quantized.numpy()
          y_np = y_var.numpy()
          print(y_np.shape, y_quantized_np.shape)
          # (2, 6, 10, 10), (2, 6, 10, 10)
    """

    def __init__(self,
                 layer,
                 weight_bits=8,
                 activation_bits=8,
                 moving_rate=0.9,
                 weight_quantize_type='abs_max',
                 activation_quantize_type='abs_max',
                 weight_pre_layer=None,
                 act_pre_layer=None,
                 weight_quant_layer=None,
                 act_quant_layer=None):
        r"""
        Constructor.

        The arguments are the same as ImperativeQuantAware.
        """
        super(QuantizedConv2DTranspose, self).__init__()
        # For Conv2DTranspose
        self._groups = getattr(layer, '_groups')
        self._stride = getattr(layer, '_stride')
        self._padding = getattr(layer, '_padding')
        self._output_padding = getattr(layer, 'output_padding')
        self._dilation = getattr(layer, '_dilation')
        self._data_format = getattr(layer, '_data_format')
        self.weight = getattr(layer, 'weight')
        self.bias = getattr(layer, 'bias')
        # For FakeQuant
        self._conv2d_transpose_quant_axis = 1
        if weight_quant_layer is not None:
            self._fake_quant_weight = weight_quant_layer()
        else:
            self._fake_quant_weight = get_fake_quant_type(
                weight_quantize_type,
                name=self.weight.name,
                moving_rate=moving_rate,
                quant_bits=weight_bits,
                dtype=self._dtype,
                quant_on_weight=True,
                channel_num=self.weight.shape[
                    self._conv2d_transpose_quant_axis],
                quant_axis=self._conv2d_transpose_quant_axis)
        if act_quant_layer is not None:
            self._fake_quant_input = act_quant_layer()
        else:
            self._fake_quant_input = get_fake_quant_type(
                activation_quantize_type,
                name=layer.full_name(),
                moving_rate=moving_rate,
                quant_bits=activation_bits,
                dtype=self._dtype,
                quant_on_weight=False)

        self._act_preprocess = act_pre_layer(
        ) if act_pre_layer is not None else None
        self._weight_preprocess = weight_pre_layer(
        ) if weight_pre_layer is not None else None

    def forward(self, input, output_size=None):
        if self._act_preprocess is not None:
            input = self._act_preprocess(input)
        quant_input = self._fake_quant_input(input)

        weight = self.weight
        if self._weight_preprocess is not None:
            weight = self._weight_preprocess(self.weight)
        quant_weight = self._fake_quant_weight(weight)

        if output_size is None:
            output_padding = self._output_padding
        else:
            output_padding = 0

        return F.conv2d_transpose(
            quant_input,
            quant_weight,
            bias=self.bias,
            padding=self._padding,
            output_padding=output_padding,
            stride=self._stride,
            dilation=self._dilation,
            groups=self._groups,
            output_size=output_size,
            data_format=self._data_format)


class QuantizedLinear(layers.Layer):
    """
    The computational logic of QuantizedLinear is the same with Linear.
    The only difference is that its inputs are all fake quantized.
    """

    def __init__(self,
                 layer,
                 weight_bits=8,
                 activation_bits=8,
                 moving_rate=0.9,
                 weight_quantize_type='abs_max',
                 activation_quantize_type='abs_max',
                 weight_pre_layer=None,
                 act_pre_layer=None,
                 weight_quant_layer=None,
                 act_quant_layer=None):
        super(QuantizedLinear, self).__init__()
        # For Linear
        self.weight = getattr(layer, 'weight')
        self.bias = getattr(layer, 'bias')
        self.name = getattr(layer, 'name')
        # For FakeQuant
        self._linear_quant_axis = 1

        if weight_quant_layer is not None:
            self._fake_quant_weight = weight_quant_layer()
        else:
            self._fake_quant_weight = get_fake_quant_type(
                weight_quantize_type,
                name=self.weight.name,
                moving_rate=moving_rate,
                quant_bits=weight_bits,
                dtype=self._dtype,
                quant_on_weight=True,
                channel_num=self.weight.shape[self._linear_quant_axis],
                quant_axis=self._linear_quant_axis)

        if act_quant_layer is not None:
            self._fake_quant_input = act_quant_layer()
        else:
            self._fake_quant_input = get_fake_quant_type(
                activation_quantize_type,
                name=layer.full_name(),
                moving_rate=moving_rate,
                quant_bits=activation_bits,
                dtype=self._dtype,
                quant_on_weight=False)

        self._act_preprocess = act_pre_layer(
        ) if act_pre_layer is not None else None
        self._weight_preprocess = weight_pre_layer(
        ) if weight_pre_layer is not None else None

    def forward(self, input):
        if self._act_preprocess is not None:
            input = self._act_preprocess(input)
        quant_input = self._fake_quant_input(input)

        weight = self.weight
        if self._weight_preprocess is not None:
            weight = self._weight_preprocess(self.weight)
        quant_weight = self._fake_quant_weight(weight)

        out = F.linear(
            x=quant_input, weight=quant_weight, bias=self.bias, name=self.name)
        return out



