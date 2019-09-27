import networkx
import numpy
import scipy
import scipy.stats


class BayesianNetwork:
    NAME = "BayesianNetwork"

    def __init__(self, variables, *args, **kwargs):
        self.__variables = numpy.array(variables)
        self.__varidx_map = {e: i for i, e in enumerate(self.vars)}
        self.__p = len(variables)
        self.__prob = .5
        self.__acceptance_rate = .5
        if "prob" in kwargs:
            self.__prob = kwargs["prob"]
        if "acceptance_rate" in kwargs:
            self.__acceptance_rate = kwargs["acceptance_rate"]
        self.__runif = scipy.stats.uniform.rvs

    @property
    def vars(self):
        return self.__variables

    @property
    def p(self):
        return self.__p

    @property
    def name(self):
        return BayesianNetwork.NAME

    def logp(self, value):
        return 0

    def _repr_latex_(self, name=None, dist=None):
        name = r'\text{%s}' % name
        return r'${} \sim \text{{BayesianNetwork}}(\dots)$'.format(name)

    def as_graph(self, adj):
        graph = networkx.from_numpy_array(
          adj, create_using=networkx.DiGraph)
        graph = networkx.relabel_nodes(
          graph, {i: e for i, e in enumerate(self.vars)})
        return graph

    def posterior_sample(self, adj):
        new_adj = self._random(adj)
        

    def random(self, point=None):
        if point is not None:
            return self._mc_random(point.copy())
        return self._random()

    def _random(self):
        adj = numpy.zeros(shape=(self.__p, self.__p), dtype=numpy.int8)
        for i in range(0, self.p - 1):
            for j in range(i + 1, self.p):
                if self.__runif() <= self.__prob:
                    adj[i, j] = 1
        gm = numpy.random.permutation(self.vars)
        idxs = numpy.array([self.__varidx_map[g] for g in gm])
        adj = adj[idxs, :]
        adj = adj[:, idxs]
        return adj

    def _mc_random(self, adj):
        ch = numpy.random.choice(["reverse"])
        if ch == "remove":
            return self._remove_edge(adj)
        if ch == "add":
            return self._add_edge(adj)
        if ch == "reverse":
            return self._reverse_edge(adj)
        return adj

    def _remove_edge(self, adj):
        args = numpy.argwhere(adj == 1)
        idx = numpy.random.choice(range(args.shape[0]))
        adj[args[idx][0], args[idx][1]] = 0
        return adj

    def _add_edge(self, adj):
        args = numpy.argwhere(adj == 0)
        for i, j in args:
            adj[i, j] = 1
            if networkx.is_directed_acyclic_graph(self.as_graph(adj)):
                return adj
            adj[i, j] = 0
        return adj

    def _reverse_edge(self, adj):
        args = numpy.argwhere(adj == 1)
        for i, j in args:
            if adj[i, j] == 1 and adj[j, i] == 0:
                adj[i, j], adj[j, i] = 0, 1
            if networkx.is_directed_acyclic_graph(self.as_graph(adj)):
                return adj
            adj[i, j], adj[j, i] = 1, 0
        return adj
