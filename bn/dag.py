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
        self.__varidx_map = bidict({e.name: i for i, e in enumerate(self.vars)})
        self.__adj = adj

    @property
    def var_map(self):
        return self.__varidx_map

    @property
    def adj(self):
        return self.__adj

    @adj.setter
    def adj(self, value):
        self.__adj = value
        for i, v in enumerate(self.vars):
            par_idxs = np.argwhere(value[:, i] == 1).flatten()
            parents = self.vars[par_idxs]
            v.update_lpd(parents)


    @property
    def vars(self):
        return self.__vars

    @property
    def n_var(self):
        return self.__n_var

    @property
    def varnames(self):
        return self.__varnames

