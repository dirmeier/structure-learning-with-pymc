import numpy

import networkx
import numpy as np
import pandas as pd
import scipy.stats
from bidict import bidict

from bn.variable import Variable


class BayesianNetwork:
    NAME = "BayesianNetwork"

    def __init__(self, variables, adj):
        if not isinstance(variables, list):
            raise TypeError()
        if any(not isinstance(x, Variable) for x in variables):
            raise TypeError()

        self.__vars = np.array(variables)
        self.__varnames = [x.name for x in self.vars]
        self.__n_var = len(variables)
        self.__varidx_map = bidict({e.name: i for i, e in enumerate(self.vars)})
        self.__adj = adj

    @property
    def var_map(self):
        return self.__varidx_map

    @property
    def vars(self):
        return self.__vars

    @property
    def n_var(self):
        return self.__n_var

    @property
    def name(self):
        return BayesianNetwork.NAME

    @property
    def varnames(self):
        return self.__varnames

    def sample_data(self, n=1):
        df = pd.DataFrame(
          np.empty((n, self.n_var), dtype=np.str),
          columns=self.varnames)
        topo = [x for x in networkx.topological_sort(self.as_graph(self.__adj))]
        for i in range(n):
            sample = df.loc[i]
            for j, t in enumerate(topo):
                sample[self.var_map[t.name]] = t.sample(sample)
        return df

    def as_graph(self, adj):
        graph = networkx.from_numpy_array(
          adj, create_using=networkx.DiGraph)
        graph = networkx.relabel_nodes(
          graph, {i: e for i, e in enumerate(self.vars)})
        return graph
