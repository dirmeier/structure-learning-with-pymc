import numpy.random
import scipy.stats

from bn.util import expand_grid


class Variable:
    choose_from = numpy.random.choice
    rdirichlet = scipy.stats.dirichlet.rvs

    def __init__(self, name, domain, lpd=None):
        self.__name = name
        self.__domain = domain.copy()
        self.__encoding = {e: i for i, e in enumerate(self.domain)}

        self.__lpd = lpd
        self.__encoded_lpd = None
        self.__parents = self._parents()

    def __str__(self):
        return '<' + self.name + '>'

    def __repr__(self):
        return self.__str__()

    @property
    def name(self):
        return self.__name

    @property
    def domain(self):
        return self.__domain

    @property
    def lpd(self):
        return self.__lpd

    @property
    def lpd_wide(self):
        return self.__lpd.pivot_table(
          values='probability',
          index=self.__parents,
          columns=self.name)

    @staticmethod
    def rdir(n):
        return Variable.rdirichlet(alpha=numpy.repeat(1, n), size=1)[0]

    def encode(self, var):
        if isinstance(var, numpy.ndarray):
            return numpy.array([self.__encoding[x] for x in var])
        else:
            return self.__encoding[var]

    def sample(self, conditional):
        return self._sample(self.__lpd.copy(), conditional)

    def sample_encoded(self, conditional):
        if self.__encoded_lpd is None:
            val = self._sample(self.__lpd.copy(), conditional)
            return self.encode(val)
        return self._sample(self.__encoded_lpd.copy(), conditional)

    def _sample(self, loc, conditional):
        if len(self.__parents) != 0:
            for p in self.__parents:
                loc = loc[loc[p] == conditional[p]]
        var = Variable.choose_from(
          loc[self.name].values, p=loc["probability"].values)
        return var

    def _update_lpd(self, parents):
        if len(parents) == 0:
            ct = expand_grid({self.name: self.domain})
            ct['probability'] = self.rdir(ct.shape[0])
        else:
            grps = [p.name for p in parents]
            ct = expand_grid({p.name: p.domain for p in parents + [self]})
            ct['probability'] = self.rdir(ct.shape[0])
            ct['probability'] = ct.groupby(grps, group_keys=False) \
                .apply(lambda x: x.probability / x.probability.sum())

        self.__lpd = ct
        self.__encoded_lpd = self._encode_lpd(parents)
        self.__parents = self._parents()

    def _parents(self):
        return list(filter(
          lambda x: x not in [self.name, 'probability'], self.__lpd.columns))

    def _encode_lpd(self, parents):
        lpd = self.lpd.copy()
        for p in parents + [self]:
            lpd[p.name] = p.encode(lpd[p.name].values)
        return lpd



