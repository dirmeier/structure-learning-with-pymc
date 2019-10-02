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
        point['dag'], _ = self.random(adj)
        return point

    def random(self, point):
        new_point = self.propose(point.copy())
        joint_prime = self._jointlogp(new_point)
        joint = self._jointlogp(point)

        logp_prime_proposal = self._logp_proposal(point, new_point)
        logp_proposal = self._logp_proposal(new_point, point)

        ac = np.minimum(
          0, joint_prime + logp_proposal - joint - logp_prime_proposal)
        if ac > np.log(StructureMCMC.runif(size=1)):
            return new_point, joint_prime
        else:
            return point, joint

    def _jointlogp(self, point):
        p_adj = self.__var.distribution.logp(point)
        marg_lik = self._log_evidence(self.__data, point)
        joint = p_adj + marg_lik
        return joint

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

    def _logp_proposal(self, frm, to):
        adds = self._valid_add_moves(frm)
        removes = self._valid_remove_moves(frm)
        reverses = self._valid_reverse_moves(frm)

        n_edges_from = len(removes)
        n_edges_to = len(np.argwhere(to == 1))

        fc = 0
        if len(adds) != 0:
            fc += 1
        if len(reverses) != 0:
            fc += 1
        if len(removes) != 0:
            fc += 1

        # remove an edge
        if n_edges_from > n_edges_to:
            ret = 1 / (len(removes) * fc)
        # reverse edge
        elif n_edges_from == n_edges_to:
            ret = 1 / (len(reverses) * fc)
        # add an edge:
        else:
            ret = 1 / (len(adds) * fc)
        return np.log(ret)

    def propose(self, point):
        adds = self._valid_add_moves(point)
        removes = self._valid_remove_moves(point)
        reverses = self._valid_reverse_moves(point)
        ch = []
        if len(adds) != 0:
            ch += ["add"]
        if len(reverses) != 0:
            ch += ["reverse"]
        if len(removes) != 0:
            ch += ["remove"]
        ch = np.random.choice(ch)
        if ch == "remove":
            return self._remove_edge(point, removes)
        elif ch == "add":
            return self._add_edge(point, adds)
        elif ch == "reverse":
            return self._reverse_edge(point, reverses)
        return point

    def _valid_add_moves(self, adj):
        valid_moves = []
        args = np.argwhere(adj == 0)
        for i, j in args:
            adj[i, j] = 1
            if networkx.is_directed_acyclic_graph(self.as_graph(adj)):
                valid_moves += [[i, j]]
            adj[i, j] = 0
        return np.array(valid_moves)

    def _valid_remove_moves(self, adj):
        return np.argwhere(adj == 1)

    def _valid_reverse_moves(self, adj):
        valid_moves = []
        args = np.argwhere(adj == 1)
        for i, j in args:
            if adj[i, j] == 1 and adj[j, i] == 0:
                adj[i, j], adj[j, i] = 0, 1
                if self.is_dag(adj):
                    valid_moves += [[i, j]]
                adj[i, j], adj[j, i] = 1, 0
        return np.array(valid_moves)

    def _remove_edge(self, adj, moves):
        idx = np.random.choice(range(moves.shape[0]))
        adj[moves[idx][0], moves[idx][1]] = 0
        return adj

    def _add_edge(self, adj, moves):
        idx = np.random.choice(range(moves.shape[0]))
        adj[moves[idx][0], moves[idx][1]] = 1
        return adj

    def _reverse_edge(self, adj, moves):
        idx = np.random.choice(range(moves.shape[0]))
        adj[moves[idx][0], moves[idx][1]] = 0
        adj[moves[idx][1], moves[idx][0]] = 1
        return adj

    def is_dag(self, adj):
        return networkx.is_directed_acyclic_graph(self.as_graph(adj))

    def as_graph(self, adj):
        graph = networkx.from_numpy_array(
          adj, create_using=networkx.DiGraph)
        graph = networkx.relabel_nodes(
          graph, {i: e for i, e in enumerate(self.__var.distribution.vars)})
        return graph
