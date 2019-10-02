import numpy as np
import scipy.stats
from pymc3 import Discrete
from pymc3.model import FreeRV

from bn.dag import DAG
from bn.variable import Variable


class DAGPrior(Discrete, DAG):
    NAME = "DAGPrior"
    runif = scipy.stats.uniform.rvs

    def __init__(self, variables, *args, **kwargs):
        if any(not isinstance(x, Variable) for x in variables):
            raise TypeError()
        DAG.__init__(self, variables=variables, adj=None)
        Discrete.__init__(self,
                          shape=(self.n_var, self.n_var),
                          *args, **kwargs)

        self.__log_c = np.log(kwargs.get("c", 1))
        self.__prob = kwargs.get("prob", .5)
        np.random.seed(23)
        self.mode = self.random()

    @property
    def name(self):
        return DAG.NAME

    @property
    def dag(self):
        return self.__dag

    def logp(self, value):
        if isinstance(value, FreeRV):
            return 0
        return self.__log_c * len(np.argwhere(value == 1))

    @staticmethod
    def _repr_latex_(name=None, dist=None):
        name = r'\text{%s}' % name
        return r'${} \sim \text{{DAG}}(\dots)$'.format(name)

    def random(self, point=None, size=None):
        if size is None:
            size = 1
        adjs = np.zeros(shape=(size, self.n_var, self.n_var))
        for i in range(size):
            for j in range(0, self.n_var - 1):
                for k in range(j + 1, self.n_var):
                    if DAGPrior.runif() <= self.__prob:
                        adjs[i, j, k] = 1
            gm = np.random.permutation(self.vars)
            idxs = np.array([self.var_map[g.name] for g in gm])
            adjs[i] = adjs[i, idxs, :]
            adjs[i] = adjs[i, :, idxs]
        return adjs if size > 1 else adjs[0]
