import networkx
import numpy as np
import pandas as pd
from pymc3 import Discrete
from pymc3.model import FreeRV

from bn.dag import DAG


class BayesianNetwork(Discrete):
    NAME = "BayesianNetwork"

    def __init__(self, dag, *args, **kwargs):
        if not isinstance(dag, DAG) and not isinstance(dag, FreeRV):
            raise TypeError(
              "'dag' argument must be either DAG or FreeRV(DAGPrior)")
        self.__dag = dag
        super(BayesianNetwork, self).__init__(shape=self.n_var, *args, **kwargs)
        self.mode = np.repeat(0, self.n_var)

    def logp(self, value):
        return 0

    @property
    def adj(self):
        if isinstance(self.dag, DAG):
            return self.dag.adj
        return self.dag.distribution.adj

    @property
    def dag(self):
        return self.__dag

    @property
    def var_map(self):
        if isinstance(self.dag, DAG):
            return self.dag.var_map
        return self.dag.distribution.var_map

    @property
    def vars(self):
        if isinstance(self.dag, DAG):
            return self.dag.vars
        return self.dag.distribution.vars

    @property
    def n_var(self):
        if isinstance(self.dag, DAG):
            return self.dag.n_var
        return self.dag.distribution.n_var

    @property
    def varnames(self):
        if isinstance(self.dag, DAG):
            return self.dag.varnames
        return self.dag.distribution.varnames

    @property
    def name(self):
        return BayesianNetwork.NAME

    def random(self, point=None, size=None):
        if size is None:
            size = 1
        df = pd.DataFrame(np.empty((size, self.n_var), dtype=np.str),
                          columns=self.varnames)

        if isinstance(self.dag, DAG):
            topo = [x for x in networkx.topological_sort(self.as_graph())]
            for i in range(size):
                sample = df.loc[i]
                for j, t in enumerate(topo):
                    sample[self.var_map[t.name]] = t.sample(sample)
        elif isinstance(self.dag, FreeRV):
            for i in range(size):
                adj = self.dag.random()
                self.dag.distribution.adj = adj
        return df

    def as_graph(self):
        graph = networkx.from_numpy_array(
          self.adj, create_using=networkx.DiGraph)
        graph = networkx.relabel_nodes(
          graph, {i: e for i, e in enumerate(self.vars)})
        return graph
