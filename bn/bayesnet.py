import networkx
import numpy
import scipy.stats

from bn.variable import Variable


class BayesianNetwork:
    NAME = "BayesianNetwork"
    runif = scipy.stats.uniform.rvs

    def __init__(self, variables, adj=None, **kwargs):
        if not isinstance(variables, list):
            raise TypeError()
        if any(not isinstance(x, Variable) for x in variables):
            raise TypeError()

        self.__vars = numpy.array(variables)
        self.__n_var = len(variables)
        self.__varidx_map = {e.name: i for i, e in enumerate(self.vars)}
        self.__adj = adj

        self.__prob = kwargs.get("prob", .5)
        self.__c = kwargs.get("c", .9)
        self.__acceptance_rate = kwargs.get("acceptance_rate", .5)

    @property
    def vars(self):
        return self.__vars

    @property
    def n_var(self):
        return self.__n_var

    @property
    def name(self):
        return BayesianNetwork.NAME

    def logp(self, value):
        return 0

    def _repr_latex_(self, name=None, dist=None):
        name = r'\text{%s}' % name
        return r'${} \sim \text{{BayesianNetwork}}(\dots)$'.format(name)

    def sample_data(self, n=1):
        topo = [x for x in networkx.topological_sort(self.as_graph(self.__adj))]
        print(topo)


    def as_graph(self, adj):
        graph = networkx.from_numpy_array(
          adj, create_using=networkx.DiGraph)
        graph = networkx.relabel_nodes(
          graph, {i: e for i, e in enumerate(self.vars)})
        return graph

    def posterior_sample(self, adj):
        new_adj = self.random(adj)
        p_new_adj = self.log_prior_probabilty(new_adj)
        lik = self.logevidence(adj)

    def log_prior_probabilty(self, adj):
        return self.__c * len(numpy.argwhere(adj == 1))

    def logevidence(self, adj):
        evidence = 0
        for i, e in enumerate(self.vars):
            parents = numpy.argwhere(adj[:, i] == 1)[0]
            s = 2
        k = 2

    def random(self, point=None):
        if point is not None:
            return self._mc_random(point.copy())
        return self._random()

    def _random(self):
        adj = numpy.zeros(shape=(self.n_var, self.n_var), dtype=numpy.int8)
        for i in range(0, self.n_var - 1):
            for j in range(i + 1, self.n_var):
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
