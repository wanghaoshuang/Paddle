# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from collections import OrderedDict
from .... import io
from .... import core
from .... import compiler
from ....data_feeder import DataFeeder
from ....framework import Program
from ....framework import IrGraph
from ....framework import IrVarNode
from ....framework import IrOpNode
from ....framework import program_guard
from ....framework import Parameter
from ....framework import Variable
from ....executor import Executor
import copy
from collections import Iterable
import numpy as np
import pickle
import os

__all__ = [
    'Var',
    'Op',
    'Graph',
]


class Var(object):
    def __init__(self, ir_var_node, ir_graph):
        assert isinstance(
            ir_var_node,
            IrVarNode), 'ir_var_node must be the instance of IrVarNode'
        self.ir_var_node = ir_var_node
        self.ir_graph = ir_graph

    @property
    def name(self):
        return self.ir_var_node.name

    @property
    def shape(self):
        return self.ir_var_node.shape

    def set_shape(self, shape):
        return self.ir_var_node.set_shape(shape)

    @property
    def input_ops(self):
        return [Op(in_op, self.ir_graph) for in_op in self.ir_var_node.inputs]

    @property
    def output_ops(self):
        return [
            Op(out_op, self.ir_graph) for out_op in self.ir_var_node.outputs
        ]


class Param(Var):
    def __init__(ir_var_node):
        self.ir_var_node = ir_var_node


OPTIMIZER_OPS = [
    'momentum',
    'lars_momentum',
    'adagrad',
    'adam',
    'adamax',
    'decayed_adagrad',
    'adadelta',
    'rmsprop',
]


class Op(object):
    def __init__(self, ir_op_node, ir_graph):
        assert isinstance(
            ir_op_node, IrOpNode), 'ir_op_node must be the instance of IrOpNode'
        self.ir_op_node = ir_op_node
        self.ir_graph = ir_graph

    @property
    def input_var_names(self):
        return [in_var.name for in_var in self.input_vars()]

    @property
    def output_var_names(self):
        return [out_var.name for out_var in self.output_vars()]

    @property
    def input_vars(self):
        return [
            Var(in_ir_var, self.ir_graph)
            for in_ir_var in self.ir_op_node.inputs
        ]

    @property
    def output_vars(self):
        return [
            Var(out_ir_var, self.ir_graph)
            for out_ir_var in self.ir_op_node.outputs
        ]

    @property
    def idx(self):
        return self.ir_op_node.id()

    @property
    def type(self):
        return self.ir_op_node.op().type()

    def infer_shape(self):
        return self.ir_op_node.op().infer_shape(self.ir_op_node.op().block())

    def is_bwd_op(self):
        return self.type.endswith('_grad')

    def is_opt_op(self):
        return op.type in OPTIMIZER_OPS

    def vars_of_input(self, input_name):
        return [
            Var(self.ir_graph.var_node(in_var), self.ir_graph)
            for in_var in self.ir_op_node.input(input_name)
        ]

    def var_names_of_input(self, input_name):
        return self.ir_op_node.input(input_name)

    def set_attr(self, key, value):
        self.ir_op_node.set_attr(key, value)


class Graph(object):
    def __init__(self,
                 program,
                 scope,
                 in_nodes=[],
                 out_nodes=[],
                 place=None,
                 for_test=False):
        super(Graph, self).__init__()
        self.for_test = for_test
        self._program = Program() if program is None else program
        self.param_names = []
        for block in program.blocks:
            self.param_names += [param.name for param in block.all_parameters()]
        self.param_names = set(self.param_names)

        self._data_feeder = DataFeeder(
            in_nodes.values(), place, program=program)
        if for_test:
            self.ir_graph = IrGraph(core.Graph(program.desc), for_test=for_test)
        else:
            self.ir_graph = None
        self.compiled_graph = None
        self._scope = scope
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes
        self._attrs = collections.OrderedDict()

    @property
    def data_feeder(self):
        return self._data_feeder

    def re_compile(self, for_parallel=True):

        if self.for_test:
            program = self.ir_graph.graph
            loss = None
        else:
            program = self.program
            loss = self.out_nodes['loss']
        if for_parallel:
            self.compiled_graph = compiler.CompiledProgram(
                program).with_data_parallel(loss_name=loss)
        else:
            self.compiled_graph = compiler.CompiledProgram(program)

    def has(self, attr_name):
        return attr_name in self._attrs

    def set(self, attr_name, attr):
        if not has(attr_name):
            self._attrs[attr_name] = attr
        else:
            raise ValueError("{} attr already set in the graph.".format(
                attr_name))

    def get(self, attr_name):
        if has(attr_name):
            return self._attrs[attr_name]
        else:
            raise ValueError("{} attr not registered in the graph.".format(
                attr_name))

    def all_parameters(self):
        return [self.var(param_name) for param_name in self.param_names]

    def all_vars(self):
        return [
            Var(var, self.ir_graph) for var in self.ir_graph.all_var_nodes()
        ]

    @property
    def scope(self):
        return self._scope

    @property
    def ops(self):
        return [
            Op(op_node, self.ir_graph)
            for op_node in self.ir_graph.all_op_nodes()
        ]

    def var(self, name):
        return Var(self.ir_graph.var_node(name), self.ir_graph)

    def clone(self):
        # TODO(wanghaoshuang@baidu.com): use clone function of IrGraph
        return Graph(self.program.clone(), self.scope,
                     copy.deepcopy(self.in_nodes),
                     copy.deepcopy(self.out_nodes))

    def pre_ops(self, op):
        """
        Get all the previous operators of target operator.
        args:
            op: Target operator. It should be an instance of class slim.core.Op.
        return: A list of operators.
        """
        ops = []
        for in_var in op.input_vars:
            for in_op in in_var.input_ops:
                ops.append(in_op)

    def next_ops(self, op):
        ops = []
        for out_var in op.output_vars:
            for out_op in out_var.output_ops:
                ops.append(out_op)
        return ops

    def get_param_by_op(self, op):
        params = []
        for in_var in op.input_vars:
            if in_var.name in self.param_names:
                params.append(in_var)
        return params

    def flops(self):
        ret = 0
        b_vars = {}
        for var in self.all_vars():
            b_vars[var.name] = var
        for op in self.ops:
            if op.type in ['conv2d', 'depthwise_conv2d', 'mul']:
                _, _, _, flop = _count_shape_params_flops(b_vars, op)
                ret += flop
        return ret

    def numel_params(self):
        ret = 0
        for param in self.all_parameters():
            ret += np.product(param.shape)
        return ret

    def serialize(self):
        data = {}
        data['program'] = self.program
        data['ir_graph'] = ir_graph
        data['in_nodes'] = self.in_nodes
        data['out_nodes'] = self.out_nodes
        data['attrs'] = self._attrs
        return pickle.dumps(data)

    def deserialize(self, s):
        data = pickle.loads(s)
        self._program = data['program']
        self.ir_graph = data['ir_graph']
        self.in_nodes = data['in_nodes']
        self.out_nodes = data['out_nodes']
        self._attrs = data['attrs']

    def get_optimize_graph(self, optimizer, place):
        """
        Append backward operators and optimize operators to graph.
        """
        main_program = self.ir_graph.to_program()
        print("get_optimize_graph")

        graph = Graph(
            main_program,
            self.scope,
            in_nodes=self.in_nodes,
            out_nodes=self.out_nodes,
            place=place,
            for_test=False)

        startup_program = Program()
        with program_guard(
                main_program=main_program, startup_program=startup_program):
            target_name = None
            if 'loss' in self.out_nodes:
                target_name = self.out_nodes['loss']
            elif 'cost' in self.out_nodes:
                target_name = self.out_nodes['cost']
            target = main_program.global_block().var(target_name)
            optimizer.minimize(target)

        exe = Executor(place)
        exe.run(program=startup_program, scope=self.scope)

        return graph

    @property
    def program(self):
        return self._program

    def save_persistables(self, path, exe):
        with scope_guard(self.scope):
            io.save_persistables(exe.exe, path, main_program=self.program)

    def load_persistables(self, path, exe):
        def if_exist(var):
            return os.path.exists(os.path.join(path, var.name))

        io.load_vars(
            exe.exe, path, main_program=self.program, predicate=if_exist)
        self.update_param_shape()
        self.update_groups_of_conv()

    def update_param_shape(self):
        if self.ir_graph is not None:
            for param in self.all_parameters():
                tensor_shape = np.array(
                    self.scope.find_var(param.name()).get_tensor()).shape
                param.set_shape(tensor_shape)
        # program is used while this graph is for training.
        if not self.for_test:
            for param in self.program.global_block().all_parameters():
                tensor_shape = np.array(
                    self.scope.find_var(param.name).get_tensor()).shape
                param.desc.set_shape(tensor_shape)

    def infer_shape(self):
        if self.ir_graph is not None:
            for op in self.ops:
                if op.type != 'conditional_block':
                    op.infer_shape()
        # program is used while this graph is for training.
        if not self.for_test:
            for op in self.program.global_block().ops:
                if op.type != 'conditional_block':
                    op.desc.infer_shape()

    def update_groups_of_conv(self):
        if self.ir_graph is not None:
            for op in self.ops:
                if op.type == 'depthwise_conv2d':
                    op.set_attr('groups',
                                op.vars_of_input('Filter')[0].shape()[0])
        # program is used while this graph is for training.
        if not self.for_test:
            for op in self.program.global_block().ops:
                if op.type == 'depthwise_conv2d':
                    op.desc._set_attr('groups',
                                      self.program.global_block().var(
                                          op.input('Filter')[0]).shape[0])


def _count_shape_params_flops(b_vars, one_op):
    '''
    Args:
        b_vars: all vars of one block
        one_op: one operator to count
    Returns:
        in_data_shape: one operator's input data shape
        out_data_shape: one operator's output data shape
        PARAMs: one operator's PARAMs 
        FLOPs: : one operator's FLOPs
    '''
    if one_op.type in ['conv2d', 'depthwise_conv2d']:
        k_arg_shape = b_vars[one_op.var_names_of_input("Filter")[0]].shape
        in_data_shape = b_vars[one_op.var_names_of_input("Input")[0]].shape
        out_data_shape = b_vars[one_op.var_names_of_output("Output")[0]].shape
        c_out, c_in, k_h, k_w = k_arg_shape
        _, c_out_, data_h, data_w = out_data_shape
        #        assert c_out == c_out_, 'shape error!'
        k_groups = one_op.attr("groups")
        kernel_ops = k_h * k_w * (c_in / k_groups)
        # keras's conv use bias defaultly
        # bias_ops = 0 if one_op.input("Bias") == [] else 1
        bias_ops = 0  # for test
        PARAMs = c_out * (kernel_ops + bias_ops)
        FLOPs = 2 * data_h * data_w * c_out * (kernel_ops + bias_ops)

    elif one_op.type == 'pool2d':
        in_data_shape = b_vars[one_op.var_names_of_input("X")[0]].shape
        out_data_shape = b_vars[one_op.var_names_of_output("Out")[0]].shape
        _, c_out, data_h, data_w = out_data_shape
        k_size = one_op.attr("ksize")
        PARAMs = 0
        FLOPs = data_h * data_w * c_out * (k_size[0]**2)

    elif one_op.type == 'mul':
        k_arg_shape = b_vars[one_op.var_names_of_input("Y")[0]].shape
        in_data_shape = b_vars[one_op.var_names_of_input("X")[0]].shape
        out_data_shape = b_vars[one_op.var_names_of_output("Out")[0]].shape
        # TODO: fc has mul ops
        # add attr to mul op, tell us whether it belongs to 'fc'
        # this's not the best way
        if 'fc' not in one_op.var_names_of_output("Out")[0]:
            return None
        k_in, k_out = k_arg_shape
        # bias in sum op
        PARAMs = k_in * k_out + 1
        FLOPs = k_in * k_out

    elif one_op.type in ['relu', 'sigmoid']:
        in_data_shape = b_vars[one_op.var_nams_of_input("X")[0]].shape
        out_data_shape = b_vars[one_op.var_names_of_output("Out")[0]].shape
        _, c_in, data_h, data_w = in_data_shape
        PARAMs = 0
        FLOPs = data_h * data_w * c_in

    elif one_op.type == 'batch_norm':
        in_data_shape = b_vars[one_op.var_names_of_input("X")[0]].shape
        out_data_shape = b_vars[one_op.var_names_of_output("Y")[0]].shape
        _, c_in, data_h, data_w = in_data_shape
        # gamma, beta, mean, std
        PARAMs = c_in * 4
        FLOPs = data_h * data_w * c_in

    else:
        return None

    return in_data_shape, out_data_shape, PARAMs, FLOPs
