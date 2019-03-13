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

import copy
import pickle
import os
import logging
import sys
import numpy as np
from collections import OrderedDict
from collections import Iterable
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
from .graph_pass import OptimizeGraphPass

__all__ = ['GraphWrapper', ]

FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

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


class GraphWrapper(object):
    """
    It is a wrapper of paddle.fluid.framework.IrGraph with some special functions
    for paddle slim framework.
    """

    def __init__(self, program=None, in_nodes=[], out_nodes=[], for_test=False):
        """
        Args:
            program(framework.Program): The program will be converted to IrGraph in this wrapper.
            It is also used to maintain some information ommited in IrGraph.
            in_nodes(dict): A dict to indicate the input nodes of the graph.
                            The key is user-defined and human-readable name.
                            The value is the name of IrVarNode or Variable.
            out_nodes(dict): A dict to indicate the input nodes of the graph.
                            The key is user-defined and human-readable name.
                            The value is the name of IrVarNode or Variable.
            for_test: Whether the graph is used for test.
        """
        super(GraphWrapper, self).__init__()
        self.for_test = for_test
        # program is just used to init some information
        self.program = Program() if program is None else program
        self.param_names = []
        for block in program.blocks:
            self.param_names += [param.name for param in block.all_parameters()]
        self.param_names = set(self.param_names)
        self.ir_graph = IrGraph(core.Graph(program.desc), for_test=for_test)
        self.compiled_graph = None
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes
        self._attrs = OrderedDict()

    def is_parameter(self, var):
        """
        Whether the variable is parameter.
        Args:
            var(framework.IrVarNode): Target variable.
        Returns:
            bool: True if the variable is parameter.
            
        """
        assert isinstance(
            var, IrVarNode), "var should be instance of framework.IrVarNode."
        return var.name() in self.param_names

    def is_bwd_op(self, op):
        """
        Whether the operator is backward op in this graph.
        Args:
            op(framework.IrOpNode): Target operator.
        Returns:
            bool: True if the operator is backward op.
        """
        return op.op().type().endswith('_grad')

    def is_opt_op(self, op):
        """
        Whether the operator is used for optimization.
        Args:
            op(framework.IrOpNode): Target operator.
        Returns:
            bool: True if the operator is used for optimization.
        """
        return op.op().type() in OPTIMIZER_OPS

    def compile(self, for_parallel=True):
        """
        Compile the ir_graph in this wrapper to framework.CompiledProgram for next running.
        This function must be called if the ir_graph is modified.
        Args:
            for_parallel(bool): Whether the program to run in data parallel way.
        """
        target = self.ir_graph.clone().graph
        if self.for_test:
            loss = None
        else:
            loss = self.out_nodes['loss']
        if for_parallel:
            # disable memory optimize for stable training
            build_strategy = compiler.BuildStrategy()
            build_strategy.enable_inplace = False
            build_strategy.memory_optimize = False
            self.compiled_graph = compiler.CompiledProgram(
                target).with_data_parallel(
                    loss_name=loss, build_strategy=build_strategy)
        else:
            self.compiled_graph = compiler.CompiledProgram(target)

    def all_parameters(self):
        """
        Get all the parameters in this graph.
        Returns:
            list<IrVarNode>: A list of IrVarNode instances.
        """
        params = []
        for var in self.ir_graph.all_persistable_nodes():
            if var.name() in self.param_names:
                params.append(var)
        return params

    def all_vars(self):
        """
        Return all variable nodes included in the graph as a set.
        """
        return self.ir_graph.all_var_nodes()

    def all_persistables(self):
        """
        Return all persistable variable nodes included in the graph as a set.
        """
        return self.ir_graph.all_persistable_nodes()

    def ops(self):
        """
        Return all operator nodes included in the graph as a set.
        """
        return self.ir_graph.all_op_nodes()

    def var(self, name):
        """
        Return all variable nodes included in the graph as a set.
        """
        return self.ir_graph.var_node(name)

    def pre_ops(self, op):
        """
        Get all the previous operators of target operator.
        Args:
            op(IrOpNode): Target operator..
        Returns:
            list<IrOpNode>: A list of operators.
        """
        ops = []
        for in_var in op.inputs:
            for in_op in in_var.inputs:
                ops.append(in_op)
        return ops

    def next_ops(self, op):
        """
        Get all the next operators of target operator.
        Args:
            op(IrOpNode): Target operator..
        Returns:
            list<IrOpNode>: A list of operators.
        """
        ops = []
        for out_var in op.outputs:
            for out_op in out_var.outputs:
                ops.append(out_op)
        return ops

    def get_param_by_op(self, op):
        """
        Get the parameters used by target operator.
        """
        params = []
        for in_var in op.inputs:
            if in_var.name in self.param_names:
                params.append(in_var)
        return params

    def numel_params(self):
        """
        Get the number of elements in all parameters.
        """
        ret = 0
        for param in self.all_parameters():
            ret += np.product(param.shape())
        return ret

    def get_optimize_graph(self, optimizer, place, scope, no_grad_var_names=[]):
        """
        Get a new graph for training by appending some backward operators and optimization operators.

        Args:
            optimizer: The optimzier used to generate training graph.
            place: The place to run the graph.
            scope: The scope used to run the graph.
            no_grad_var_names(list<str>): Names of variables that should be ignored while computing gradients. default: [].
        """
        opt_pass = OptimizeGraphPass(
            scope,
            place,
            optimizer,
            self.out_nodes['loss'],
            self.program,
            no_grad_var_names=no_grad_var_names)
        opt_ir_graph = opt_pass.apply(self.ir_graph)

        graph = GraphWrapper(
            program=self.program,
            in_nodes=self.in_nodes,
            out_nodes=self.out_nodes,
            for_test=False)
        graph.ir_graph = opt_ir_graph
        graph.param_names = self.param_names
        return graph

    def save_persistables(self, path, exe):
        """
        Save all the persistable variables into file.
        Args:
            path(str): The path to save the persistables.
            exe(framework.Executor): The executor used to save the persistables.
        """
        io.save_persistables(exe, path, main_program=self.program)

    def load_persistables(self, path, exe):
        """
        Load the persistable variables from file.
        Args:
            path(str): The path to load the persistables.
            exe(framework.Executor): The executor used to load the persistables.
        """

        def if_exist(var):
            return os.path.exists(os.path.join(path, var.name))

        io.load_vars(exe, path, main_program=self.program, predicate=if_exist)
        self.update_param_shape()
        self.update_groups_of_conv()

    def update_param_shape(self, scope):
        """
        Update the shape of parameters in the graph according to tensors in scope.
        It is used after loading pruned parameters from file.
        """
        if self.ir_graph is not None:
            for param in self.all_parameters():
                tensor_shape = np.array(
                    scope.find_var(param.name()).get_tensor()).shape
                param.set_shape(tensor_shape)

    def update_groups_of_conv(self):
        """
        Update the groups of convolution layer according to current filters.
        It is used after loading pruned parameters from file.
        """
        if self.ir_graph is not None:
            for op in self.ops:
                if op.op().type() == 'depthwise_conv2d':
                    op.set_attr('groups',
                                self.ir_graph.var_node(op.input('Filter')[0])
                                .shape()[0])

    def flops(self):
        """
        Get the flops of current graph.
        """
        flops = 0
        ir_graph = self.ir_graph
        for op in ir_graph.all_op_nodes():
            if op.op().type() in ['conv2d', 'depthwise_conv2d']:
                op.input
                filter_shape = ir_graph.var_node(op.input("Filter")[0]).shape()
                input_shape = ir_graph.var_node(op.input("Input")[0]).shape()
                output_shape = ir_graph.var_node(op.output("Output")[0]).shape()
                c_out, c_in, k_h, k_w = filter_shape
                _, _, h_out, w_out = output_shape
                groups = op.op().attr("groups")
                kernel_ops = k_h * k_w * (c_in / groups)
                if len(op.input("Bias")) > 0:
                    with_bias = 1
                else:
                    with_bias = 0
                flops += 2 * h_out * w_out * c_out * (kernel_ops + with_bias)

            elif op.op().type() == 'pool2d':
                input_shape = ir_graph.var_node(op.input("X")[0]).shape()
                output_shape = ir_graph.var_node(op.output("Out")[0]).shape()
                _, c_out, h_out, w_out = output_shape
                k_size = op.op().attr("ksize")
                flops += h_out * w_out * c_out * (k_size[0]**2)

            elif op.op().type() == 'mul':
                x_shape = ir_graph.var_node(op.input("X")[0]).shape()
                y_shape = ir_graph.var_node(op.input("Y")[0]).shape()
                if x_shape[0] == -1:
                    x_shape[0] = 1
                flops += 2 * x_shape[0] * x_shape[1] * y_shape[1]

            elif op.op().type() in ['relu', 'sigmoid', 'batch_norm']:
                input_shape = ir_graph.var_node(op.input("X")[0]).shape()
                if input_shape[0] == -1:
                    input_shape[0] = 1
                flops += np.product(input_shape)

        return flops
