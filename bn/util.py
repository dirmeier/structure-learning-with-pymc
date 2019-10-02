import numpy
from itertools import product

import pandas as pd
from scipy.special import gamma


def expand_grid(dic):
    return pd.DataFrame(
      [row for row in product(*dic.values())],
      columns=dic.keys())


def mv_beta(alphas):
    return numpy.prod([gamma(x) for x in alphas]) / gamma(numpy.sum(alphas))
