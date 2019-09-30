import numpy.random
import scipy.stats
from sklearn.preprocessing import LabelEncoder

from bn.util import expand_grid


class Variable:
    choose_from = numpy.random.choice
    rdirichlet = scipy.stats.dirichlet.rvs
    def __init__(self, name, domain, lpd=None):
        self.__name = name
        self.__domain = sorted(domain)
        self.__encoding = LabelEncoder().fit_transform(self.domain)
        self.__lpd = lpd
        self.__parents = self._parents()

    def __str__(self):
        return '<' + self.name + '>'

    def __repr__(self):
        return self.__str__()

    def _parents(self):
        return list(filter(
          lambda x: x not in [self.name, 'probability'], self.__lpd.columns))

    @property
    def name(self):
        return self.__name

    @property
    def domain(self):
        return self.__domain

    @property
    def encoding(self):
        return self.__encoding

    @property
    def lpd(self):
        return self.__lpd

    @property
    def lpd_wide(self):
        return self.__lpd.pivot_table(
          values='probability',
          index=self.__parents,
          columns=self.name)

    def sample(self, conditional):
        loc = self.lpd
        if len(self.__parents) != 0:
            for p in self.__parents:
                loc = loc[loc[p] == conditional[p]]
            assert(loc.shape[0] == len(self.domain))
            assert (loc.shape[1] == len(self.__parents) + 2)
        return Variable.choose_from(
            loc[self.name].values, p=loc["probability"].values)

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
        self.__parents = self._parents()

    @staticmethod
    def rdir(n):
        return Variable.rdirichlet(alpha=numpy.repeat(1, n), size=1)[0]
