from sklearn.preprocessing import LabelEncoder


class Variable:
    def __init__(self, name, domain, lpd=None):
        self.__name = name
        self.__domain = sorted(domain)
        self.__encoding = LabelEncoder().fit_transform(self.domain)
        self.__lpd = lpd

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
        cols = list(self.__lpd.columns)
        cols.remove(self.name)
        cols.remove('probability')
        return self.__lpd.pivot_table(
          values='probability',
          index=cols,
          columns=self.name)

    def sample(self, conditional):
        return None
