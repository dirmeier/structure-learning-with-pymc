from sklearn.preprocessing import LabelEncoder


class Variable:
    NAME = "BayesianNetwork"

    def __init__(self, name, domain, lpd=None):
        self.__name = name
        self.__domain = sorted(domain)
        self.__encoding = LabelEncoder().fit_transform(self.domain)
        self.__lpd = lpd

    @property
    def name(self):
        return self.__name

    @property
    def domain(self):
        return self.__domain

    @property
    def encoding(self):
        return self.__encoding

    def sample(self, conditional):
        return None