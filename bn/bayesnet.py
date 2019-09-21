import networkx
import numpy
import scipy
import scipy.stats
from random import shuffle

class BayesianNetwork:
    NAME = "BayesianNetwork"

    def __init__(self, variables, *args, **kwargs):
        self.__variables = variables
        self.__p = len(variables)
        self.__prob = 0.5
        if "prob" in kwargs:
            self.__prob = kwargs["prob"]
        print(self.__prob)
        self.__runif = scipy.stats.uniform.rvs

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

    def random(self, point=None):
        next_point = numpy.zeros(shape=(self.__p, self.__p), dtype=numpy.int8)
        if point is None:
            return self._random(next_point)
        else:
            return self._mc_random(point.copy())

    def _random(self, next_point):
        for i in range(0, self.p - 1):
            for j in range(i + 1, self.p):
                if self.__runif() <= self.__prob:
                    next_point[i, j] = 1
        shuffle(self.__variables)
        graph = networkx.from_numpy_array(next_point, create_using=networkx.DiGraph)
        graph = networkx.relabel_nodes(
          graph,
          {i: e for i, e in enumerate(self.__variables)})
        return graph

    def _mc_random(self, next_point):

        ch = numpy.random.choice(["remove", "add", "reverse"])
        #if ch == "remove":
        args = numpy.argwhere(next_point == 1)
        idx = numpy.random.choice(range(args.shape[0]))
        next_point[ args[idx][0], args[idx][1] ] = 0

        graph = networkx.from_numpy_array(next_point,
                                          create_using=networkx.DiGraph)
        graph = networkx.relabel_nodes(
          graph,
          {i: e for i, e in enumerate(self.__variables)})

        return graph
