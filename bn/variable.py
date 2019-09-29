import numpy.random
from sklearn.preprocessing import LabelEncoder


class Variable:
    choose_from = numpy.random.choice

    def __init__(self, name, domain, lpd=None):
        self.__name = name
        self.__domain = sorted(domain)
        self.__encoding = LabelEncoder().fit_transform(self.domain)
        self.__lpd = lpd
        self.__parents = list(filter(
          lambda x: x not in [self.name, 'probability'],
          self.__lpd.columns))

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


