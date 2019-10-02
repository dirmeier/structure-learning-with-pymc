import numpy as np
from bidict import bidict

from bn.variable import Variable


class DAG:
    def __init__(self, variables, adj):
        if any(not isinstance(x, Variable) for x in variables):
            raise TypeError()
        self.__vars = np.array(variables)
        self.__varnames = [x.name for x in self.vars]
        self.__n_var = len(variables)
        self.__varidx_map = bidict(
          {e.name: i for i, e in enumerate(self.vars)})
        self.__adj = adj
        if adj is not None:
            self._update_lpds()

    @property
    def var_map(self):
        return self.__varidx_map

    @property
    def adj(self):
        return self.__adj

    @adj.setter
    def adj(self, value):
        self.__adj = value
        self._update_lpds()

    @property
    def vars(self):
        return self.__vars

    @property
    def n_var(self):
        return self.__n_var

    @property
    def varnames(self):
        return self.__varnames

    def _update_lpds(self):
        for i, v in enumerate(self.vars):
            v._update_lpd(self.__parents(i))

    def __parents(self, v):
        idx = self.__as_idx(v)
        if idx is None:
            return []
        par_idxs = np.argwhere(self.adj[:, idx] == 1).flatten()
        parents = list(self.vars[par_idxs])
        return parents

    def __as_idx(self, v):
        if isinstance(v, Variable):
            return self.var_map[v.name]
        elif isinstance(v, str):
            return self.var_map[v]
        return v
