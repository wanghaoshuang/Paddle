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

from ....core import CPUPlace
from ....data_feeder import DataFeeder
from ..graph import get_executor
from config import ConfigFactory

__all__ = ['Context', 'CompressPass']


class Context(object):
    """
    The context in the process of compression.
    Args:
        exe: The executor used to execute graph.
        graph: The graph to be compressed.
        scope: The scope used to execute graph.
        program_exe: The program_exe is used to execute the program
                     created for modifying the variables in scope.
    """

    def __init__(self,
                 graph,
                 place,
                 train_graph=None,
                 train_reader=None,
                 train_feeder=None,
                 eval_graph=None,
                 eval_reader=None,
                 eval_feeder=None):
        # The total number of epoches to be trained.
        self.epoch = 0
        # Current epoch
        self.epoch_id = 0
        # Current batch
        self.batch_id = 0
        self.k_v = {}

        self.graph = graph
        self.place = place
        self.train_graph = train_graph
        self.train_reader = train_reader
        self.train_feeder = train_feeder
        self.eval_graph = eval_graph
        self.eval_reader = eval_reader
        self.eval_feeder = eval_feeder

    def run_eval_graph(self):
        assert self.eval_graph is not None
        assert self.eval_reader is not None
        assert self.eval_feeder is not None
        results = []
        for data in self.eval_reader():
            feed = self.eval_feeder.feed(data)
            result = self.executor.run(self.eval_graph, feed=feed)
            results.append(result)
        return np.mean(
            np.array(results), axis=0), self.eval_graph.out_nodes.keys()

    def put(self, key, value):
        self.k_v[key] = value

    def get(self, key):
        return self.k_v.get(key)


class CompressPass(object):
    """
    The pass used to compress model.
    Args:
        place: The device used in compression.
        data_reader: The data_reader used to run graph.
        data_feeder: The data_feeder used to run graph.
        scope: The scope used to run graph.
        metrics: The metrics for evaluating model.
        epoch: The total epoches of trainning in compression.
        program_exe: The program_exe is used to execute the program
                     created for modifying the variables in scope.
    """

    def __init__(self,
                 place=None,
                 train_graph_pass=None,
                 train_reader=None,
                 train_feed_list=None,
                 eval_graph_pass=None,
                 eval_reader=None,
                 eval_feed_list=None):
        self.strategies = []
        self.epoch = 0
        self.place = CPUPlace() if place is None else place
        self.train_graph_pass = train_graph_pass
        self.train_reader = train_reader
        self.train_feed_list = train_feed_list
        self.eval_graph_pass = eval_graph_pass
        self.eval_reader = eval_reader
        self.eval_feed_list = eval_feed_list

    def add_strategy(self, strategy):
        """
        Add a strategy to current compress pass.
        Args:
            strategy: The strategy to be added into current compress pass.
        """
        self.strategies.append(strategy)
        self.epoch = max(strategy.end_epoch, self.epoch)

    def config(self, config_file):
        factory = ConfigFactory(config_file)
        self.epoch = factory.compress_pass['epoch']
        for strategy in factory.compress_pass['strategies']:
            self.add_strategy(strategy)

    def _train_one_epoch(self, context):
        if context.train_graph is None:
            print("train_graph is None; Please config train_graph_pass.")
            return
        for data in context.train_reader():
            for strategy in self.strategies:
                strategy.on_batch_begin(context)
            feed = None
            if context.train_feeder:
                feed = context.train_feeder.feed(data)
            results = self.executor.run(context.train_graph, feed=feed)
            print("epoch:{}; batch_id:{}; train results: {}".format(
                context.epoch, context.batch_id, results))
            for strategy in self.strategies:
                strategy.on_batch_end(context)
            context.batch_id += 1

    def _eval(self, context):
        result, names = context.run_eval_grap()
        print("epoch:{}; batch_id:{}; eval results: {}={}".format(
            context.epoch, context.batch_id, names, results))

    def apply(self, graph):
        """
        Compress a model.
        Args:
            graph: The target graph to be compressed.
        """
        self.executor = get_executor(graph, self.place)
        train_graph = None
        train_feeder = None
        if self.train_graph_pass is not None:
            train_graph = self.train_graph_pass.apply(graph)
            train_feeder = DataFeeder(
                feed_list=self.train_feed_list,
                place=self.place,
                program=train_graph.program)
        eval_graph = None
        eval_feeder = None
        if self.eval_graph_pass is not None:
            eval_graph = self.eval_graph_pass.apply(graph)
            eval_feeder = DataFeeder(
                feed_list=self.eval_feed_list,
                place=self.place,
                program=eval_graph.program)

        context = Context(
            graph=graph,
            place=self.place,
            train_graph=train_graph,
            train_reader=self.train_reader,
            train_feeder=train_feeder,
            eval_graph=eval_graph,
            eval_reader=self.eval_reader,
            eval_feeder=eval_feeder)

        for strategy in self.strategies:
            strategy.on_compress_begin(context)

        for epoch in range(self.epoch):

            for strategy in self.strategies:
                strategy.on_epoch_begin(context)

            self._train_one_epoch(context)

            for strategy in self.strategies:
                strategy.on_epoch_end(context)
            context.epoch_id += 1

            if epoch % self.eval_epoch == 0:
                self._eval(context)

        for strategy in self.strategies:
            strategy.on_compress_end(context)

        return context.graph
