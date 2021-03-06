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
from .... import parallel_executor
from ....data_feeder import DataFeeder
from .graph import IRGraph, ImitationGraph

__all__ = ['get_executor']


class GraphExecutor(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, place, parallel=True):
        self.place = place

    @abstractmethod
    def run(self, graph, feches=None, feed=None):
        pass


class IRGraphExecutor(GraphExecutor):
    def run(self, grah, fetches, feed=None):
        pass


class ImitationGraphExecutor(GraphExecutor):
    def __init__(self, place, parallel=True):
        super(ImitationGraphExecutor, self).__init__(place, parallel=parallel)
        self.parallel = parallel
        self.exe = None
        self.place = place

        if not parallel:
            self.exe = executor.Executor(place)

    def run(self, graph, data=None, fetches=None):
        assert isinstance(graph, ImitationGraph)
        feeder = DataFeeder(
            feed_list=graph.in_nodes.values(),
            place=self.place,
            program=graph.program)
        feed = feeder.feed(data)

        fetch_list = fetches if fetches else graph.out_nodes.values()
        #        print "fetch_list: %s" % (fetch_list, )
        if self.exe is None:
            strategy = parallel_executor.ExecutionStrategy()
            strategy.num_threads = 1
            self.exe = parallel_executor.ParallelExecutor(
                use_cuda=True,
                loss_name=graph.out_nodes['cost']
                if 'cost' in graph.out_nodes else None,
                main_program=graph.program,
                scope=graph.scope,
                exec_strategy=strategy)
        if self.parallel:
            results = self.exe.run(feed=feed, fetch_list=fetch_list)
        else:
            results = self.exe.run(graph.program,
                                   scope=graph.scope,
                                   fetch_list=fetch_list,
                                   feed=feed)
        return results


def get_executor(graph, place, parallel=True):
    if isinstance(graph, ImitationGraph):
        return ImitationGraphExecutor(place, parallel=parallel)
    if isinstance(graph, IRGraph):
        return IRGraphExecutor(place, parallel=parallel)
