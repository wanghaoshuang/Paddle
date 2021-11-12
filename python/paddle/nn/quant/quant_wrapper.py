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
import paddle
import paddle.nn as nn
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

from .quanters import get_fake_quant_type 

__all__ = [
    'QuantizedWrapper',
]

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')


class QuantizedWrapper(layers.Layer):
    def __init__(self,
                 layer,
                 weight_bits=8,
                 activation_bits=8,
                 moving_rate=0.9,
                 weight_quantize_type='abs_max',
                 activation_quantize_type='abs_max',
                 quant_parameters=["weight"],
                 weight_pre_layer=None,
                 act_pre_layer=None,
                 quant_axis=0,
                 onnx_style=True,
                 in_count=1):
        super(QuantizedWrapper, self).__init__()
        self._layer = layer
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.moving_rate = moving_rate
        self.weight_quantize_type = weight_quantize_type
        self.activation_quantize_type = activation_quantize_type
        self.weight_pre_layer = weight_pre_layer
        self.act_pre_layer = act_pre_layer
        self.quant_axis = quant_axis
        self.onnx_style = onnx_style


        self._fake_param_quanters = {}
        self._weight_preprocess = {}
        self._fake_param_quanters_layers = nn.LayerList()
        self._weight_preprocess_layers = nn.LayerList()
        for param_name in quant_parameters:
            param = getattr(layer, param_name)
            quanter = get_fake_quant_type(
                self.weight_quantize_type,
                name=param.name,
                moving_rate=self.moving_rate,
                quant_bits=self.weight_bits,
                dtype=self._dtype,
                quant_on_weight=True,
                channel_num=param.shape[self.quant_axis],
                quant_axis=self.quant_axis,
                onnx_style=True)
            self._fake_param_quanters[param_name] = quanter
            self._fake_param_quanters_layers.append(quanter)
            self._weight_preprocess[param_name] = weight_pre_layer() if weight_pre_layer is not None else None
            self._weight_preprocess_layers.append(self._weight_preprocess[param_name])

        self._fake_intput_quanters = None
        self._act_preprocess = None

        if self._act_preprocess is None:
            self._act_preprocess = nn.LayerList()
            for i in range(in_count):
                self._act_preprocess.append(self.weight_pre_layer() if self.act_pre_layer is not None else None)

        if self._fake_intput_quanters is None:
            self._fake_intput_quanters = nn.LayerList()
            for i in range(in_count):
                quanter = get_fake_quant_type(
                    self.activation_quantize_type,
                    name=self._layer.full_name() + str(i),
                    moving_rate=self.moving_rate,
                    quant_bits=self.activation_bits,
                    dtype=self._dtype,
                    quant_on_weight=False,
                    onnx_style=True)
                self._fake_intput_quanters.append(quanter)


    def forward(self, *inputs, **kwargs):
        # preprocess inputs
        if self.act_pre_layer is not None:
            inputs = [_preprocess(_in) for _in, _preprocess in zip(inputs,  self._act_preprocess)]
        # quant input
        inputs = [_quanter(_input) for _input, _quanter in zip(inputs, self._fake_intput_quanters)]
            
        # precess parameters
        if self.weight_pre_layer is not None:
            for _param_name, _precess in self._weight_preprocess.items():
                precessed_param = _precess(getattr(self._layer, _param_name))
                setattr(self._layer, _param_name, precessed_param)

        # quant parameters
        for _param_name, _quanter in self._fake_param_quanters.items():
            param = getattr(self._layer, _param_name)
            self._layer.__delattr__(_param_name)
            #setattr(_quanter, _param_name, param)
            #_quanter.add_parameter(_param_name, param)
            quanted_param = _quanter(param)
            setattr(self._layer, _param_name, quanted_param)
            self._layer.add_parameter(_param_name+"_backup", param)
            
#        print(f"layer forward....inputs len: {len(inputs)}; kwargs: {len(kwargs)}") 
        return self._layer(*inputs, **kwargs)
        
