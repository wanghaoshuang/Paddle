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

import abc
from abc import abstractmethod
from .... import executor
from ....compiler import CompiledProgram
from .... import parallel_executor
from ....data_feeder import DataFeeder
from .graph import Graph

__all__ = ['get_executor']


class GraphExecutor(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, place, parallel=True):
        self.place = place

    @abstractmethod
    def run(self, graph, feches=None, feed=None):
        pass


class CompiledGraphExecutor(GraphExecutor):
    def __init__(self, place):
        super(CompiledGraphExecutor, self).__init__(place)
        self.exe = executor.Executor(place)

    def run(self, graph, data=None, feed=None, fetches=None):
        assert isinstance(graph, Graph)
        if data is not None:
            feeder = DataFeeder(
                feed_list=graph.in_nodes.values(),
                place=self.place,
                program=graph.program)
            feed = feeder.feed(data)

        fetch_list = fetches if fetches else graph.out_nodes.values()
        program = graph.compiled_graph if graph.compiled_graph else graph.program
        results = self.exe.run(program,
                               scope=graph.scope,
                               fetch_list=fetch_list,
                               feed=feed)
        return results


def get_executor(graph, place):
    if isinstance(graph, Graph):
        return CompiledGraphExecutor(place)
