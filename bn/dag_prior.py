import networkx
import numpy as np
import pandas as pd
import scipy.stats
from bidict import bidict
from pymc3 import Discrete, Model

from bn.util import expand_grid, mv_beta
from bn.variable import Variable


class DAGPrior(Discrete):
    NAME = "DAG"
    runif = scipy.stats.uniform.rvs

    def __init__(self, variables, *args, **kwargs):
        if any(not isinstance(x, Variable) for x in variables):
            raise TypeError()
        if "adj" in kwargs:
            print(variables)
            self.__adj = kwargs.get("adj", None)
        else:
            k = 2
            #super(DAG, self).__init__(shape=self.n_var, *args, **kwargs)
        self.__vars = np.array(variables)
        self.__varnames = [x.name for x in self.vars]
        self.__n_var = len(variables)
        self.__varidx_map = bidict({e.name: i for i, e in enumerate(self.vars)})

        self.__prob = kwargs.get("prob", .5)
        self.__alpha = kwargs.get("alpha", 1)
        self.__log_c = np.log(kwargs.get("c", 1))
        self.__acceptance_rate = kwargs.get("acceptance_rate", .5)

        np.random.seed(23)
        self.mode = self.random()

    @property
    def var_map(self):
        return self.__varidx_map

    @property
    def dag(self):
        return self.__dag

    @property
    def vars(self):
        return self.__vars

    @property
    def n_var(self):
        return self.__n_var

    @property
    def name(self):
        return DAG.NAME

    def logp(self, value):
        return 0

    @staticmethod
    def _repr_latex_(name=None, dist=None):
        name = r'\text{%s}' % name
        return r'${} \sim \text{{DAG}}(\dots)$'.format(name)

    def posterior_sample(self, point):
        new_adj = self.random(point)

        p_adj = self._log_prior_prob(point)
        marg_lik = self._log_evidence(self.data, point)
        joint = np.exp(p_adj + marg_lik)

        p_adj_prime = self._log_prior_prob(new_adj)
        marg_lik_prime = self._log_evidence(self.data, new_adj)
        joint_prime = np.exp(p_adj_prime + marg_lik_prime)

        ac = np.minimum(1.0, joint_prime / joint)
        if ac > DAG.runif(size=1):
            return new_adj, joint_prime
        else:
            return point, joint

    def _log_prior_prob(self, adj):
        return self.__log_c * len(np.argwhere(adj == 1))

    def _log_evidence(self, data, adj):
        evidence = 0
        for i, v in enumerate(self.vars):
            par_idx = np.argwhere(adj[:, i] == 1).flatten()
            parents = None
            if len(par_idx) != 0:
                parents = [self.vars[p] for p in par_idx]
            evidence += self._local_evidence(v, data, parents)
        return evidence

    def _local_evidence(self, v, data, parents):
        ev = 0
        if parents is None:
            ct = pd.DataFrame({None})
        else:
            ct = expand_grid({p.name: p.domain for p in parents})
        for _, c in ct.iterrows():
            n_vc = self._counts(v, c, data, parents)
            alpha_vc = self._pseudocounts(v, parents)
            ev += np.log(mv_beta(n_vc + alpha_vc) / mv_beta(alpha_vc))
        return ev

    @staticmethod
    def _counts(v, c, data, parents):
        flt = ""
        if parents is not None:
            for p in parents:
                flt += "{} == '{}' and ".format(p.name, c[p.name])
        counts = [0] * len(v.domain)
        for i, e in enumerate(v.domain):
            e_flt = flt + "{} == '{}'".format(v.name, e)
            d = data.query(e_flt)
            counts[i] = d.shape[0]
        return np.array(counts)

    def _pseudocounts(self, v, parents):
        if parents is None:
            c_v = 1
        else:
            c_v = np.sum([len(p.domain) for p in parents])
        k_v = len(v.domain)
        alpha_vc = self.__alpha / (k_v * c_v)
        return np.repeat(alpha_vc, k_v)

    def random(self, point=None):
        if point is not None:
            return self._mc_random(point.copy())
        return self._random()

    def _random(self):
        adj = np.zeros(shape=(self.n_var, self.n_var), dtype=np.int8)
        for i in range(0, self.n_var - 1):
            for j in range(i + 1, self.n_var):
                if self.__runif() <= self.__prob:
                    adj[i, j] = 1
        gm = np.random.permutation(self.vars)
        idxs = np.array([self.__varidx_map[g] for g in gm])
        adj = adj[idxs, :]
        adj = adj[:, idxs]
        return adj

    def _mc_random(self, adj):
        n_edges = len(np.argwhere(adj == 1))
        if n_edges != 0:
            ch = np.random.choice(["reverse", "add", "remove"])
        else:
            ch = np.random.choice(["reverse", "add"])
        if ch == "remove":
            return self._remove_edge(adj)
        elif ch == "add":
            return self._add_edge(adj)
        elif ch == "reverse":
            return self._reverse_edge(adj)
        return adj

    def _remove_edge(self, adj):
        args = np.argwhere(adj == 1)
        idx = np.random.choice(range(args.shape[0]))
        adj[args[idx][0], args[idx][1]] = 0
        return adj

    def _add_edge(self, adj):
        args = np.argwhere(adj == 0)
        for i, j in args:
            adj[i, j] = 1
            if networkx.is_directed_acyclic_graph(self.as_graph(adj)):
                return adj
            adj[i, j] = 0
        return adj

    def _reverse_edge(self, adj):
        args = np.argwhere(adj == 1)
        for i, j in args:
            if adj[i, j] == 1 and adj[j, i] == 0:
                adj[i, j], adj[j, i] = 0, 1
            if networkx.is_directed_acyclic_graph(self.as_graph(adj)):
                return adj
            adj[i, j], adj[j, i] = 1, 0
        return adj
