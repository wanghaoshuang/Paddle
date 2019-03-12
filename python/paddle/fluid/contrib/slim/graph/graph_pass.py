#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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

from ....framework import IrGraph
from ....framework import program_guard
from ....executor import Executor
from ....framework import Program
from .... import core

__all__ = ['GraphPass', 'OptimizeGraphPass']


class GraphPass(object):
    """
    Base class for all graph pass.
    """

    def __init__(self):
        pass

    def apply(self, graph):
        pass


class OptimizeGraphPass(GraphPass):
    """
    Generate a graph for pruning parameters from target graph.
    """

    def __init__(self, scope, place, optimizer, loss_name):
        super(OptimizeGraphPass, self).__init__()
        self.optimizer = optimizer
        self.loss_name = loss_name
        self.scope = scope
        self.place = place

    def apply(self, graph):
        assert isinstance(graph, IrGraph)
        main_program = graph.to_program()
        startup_program = Program()
        with program_guard(
                main_program=main_program, startup_program=startup_program):
            target = main_program.global_block().var(self.loss_name)
            self.optimizer.minimize(target)
        exe = Executor(self.place)
        exe.run(program=startup_program, scope=self.scope)
        return IrGraph(core.Graph(main_program.desc), for_test=False)
