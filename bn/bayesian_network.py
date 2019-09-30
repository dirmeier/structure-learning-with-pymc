import networkx
import numpy as np
import pandas as pd
from pymc3 import Discrete

from bn.dag import DAG


class BayesianNetwork(Discrete):
    NAME = "BayesianNetwork"

    def __init__(self, dag, *args, **kwargs):
        if not isinstance(dag, DAG):
            raise TypeError()

        super(BayesianNetwork, self).__init__(shape=dag.n_var, *args, **kwargs)
        self.__dag = dag

        np.random.seed(23)
        self.mode = np.repeat(1, dag.n_var)

    def logp(self, value):
        return 0

    @property
    def adj(self):
        return self.dag.adj

    @property
    def dag(self):
        return self.__dag

    @property
    def var_map(self):
        return self.dag.var_map

    @property
    def vars(self):
        return self.dag.vars

    @property
    def n_var(self):
        return self.dag.n_var

    @property
    def name(self):
        return BayesianNetwork.NAME

    @property
    def varnames(self):
        return self.dag.varnames

    def random(self, point=None, size=None):
        if size is None:
            size = 1
        df = pd.DataFrame(
          np.empty((size, self.n_var), dtype=np.str),
          columns=self.varnames)

        # TODO: update: for every i sample a dag, update the LPDs and then sample the BN

        topo = [x for x in networkx.topological_sort(self.as_graph())]
        for i in range(size):
            sample = df.loc[i]
            for j, t in enumerate(topo):
                sample[self.var_map[t.name]] = t.sample(sample)
        return df.values

    def as_graph(self):
        graph = networkx.from_numpy_array(
          self.adj, create_using=networkx.DiGraph)
        graph = networkx.relabel_nodes(
          graph, {i: e for i, e in enumerate(self.vars)})
        return graph
