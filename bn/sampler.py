import networkx
import pymc3
import numpy as np
import scipy.stats
import pandas as pd
from pymc3.step_methods.arraystep import ArrayStep

from bn.util import expand_grid, mv_beta


class StructureMCMC(ArrayStep):
    runif = scipy.stats.uniform.rvs

    def __init__(self, vars, data, model=None, **kwargs):
        model = pymc3.modelcontext(model)
        if len(vars) != 1:
            raise ValueError("Please provide only one")

        vars = pymc3.inputvars(vars)
        self.__var = vars[0]
        self.__var_name = self.__var.name
        self.__data = data
        self.__alpha = kwargs.get("alpha", 1)
        self.__acceptance_rate = kwargs.get("acceptance_rate", .5)

        super(StructureMCMC, self).__init__(vars, [model.fastlogp])

    def step(self, point):
        adj = point['dag']
        point['dag'], _ = self.smc(adj)
        return point

    def smc(self, point):
        new_adj = self._random(point)

        p_adj = self.__var.distribution.logp(point)
        marg_lik = self._log_evidence(self.__data, point)
        joint = np.exp(p_adj + marg_lik)

        p_adj_prime = self.__var.distribution.logp(new_adj)
        marg_lik_prime = self._log_evidence(self.__data, new_adj)
        joint_prime = np.exp(p_adj_prime + marg_lik_prime)

        ac = np.minimum(1.0, joint_prime / joint)
        if ac > StructureMCMC.runif(size=1):
            return new_adj, joint_prime
        else:
            return point, joint

    def _log_evidence(self, data, adj):
        evidence = 0
        for i, v in enumerate(self.__var.distribution.vars):
            par_idx = np.argwhere(adj[:, i] == 1).flatten()
            parents = None
            if len(par_idx) != 0:
                parents = [self.__var.distribution.vars[p] for p in par_idx]
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

    def _random(self, point):
        n_edges = len(np.argwhere(point == 1))
        if n_edges != 0:
            ch = np.random.choice(["reverse", "add", "remove"])
        else:
            ch = np.random.choice(["reverse", "add"])
        if ch == "remove":
            return self._remove_edge(point)
        elif ch == "add":
            return self._add_edge(point)
        elif ch == "reverse":
            return self._reverse_edge(point)
        return point

    def _remove_edge(self, adj):
        args = np.argwhere(adj == 1)
        idx = np.random.choice(range(args.shape[0]))
        adj[args[idx][0], args[idx][1]] = 0
        return adj

    def _add_edge(self, adj):
        args = np.argwhere(adj == 0)
        for i, j in args:
            adj[i, j] = 1
            if networkx.is_directed_acyclic_graph(self._as_graph(adj)):
                return adj
            adj[i, j] = 0
        return adj

    def _reverse_edge(self, adj):
        args = np.argwhere(adj == 1)
        for i, j in args:
            if adj[i, j] == 1 and adj[j, i] == 0:
                adj[i, j], adj[j, i] = 0, 1
            if networkx.is_directed_acyclic_graph(self._as_graph(adj)):
                return adj
            adj[i, j], adj[j, i] = 1, 0
        return adj

    def _as_graph(self, adj):
        graph = networkx.from_numpy_array(
          adj, create_using=networkx.DiGraph)
        graph = networkx.relabel_nodes(
          graph, {i: e for i, e in enumerate(self.__var.distribution.vars)})
        return graph
