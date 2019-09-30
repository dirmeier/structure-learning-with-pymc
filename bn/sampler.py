import numpy as np
import pymc3
from pymc3.step_methods.arraystep import ArrayStep


class BNMetropolis(ArrayStep):
    def __init__(self, vars, model=None):
        model = pymc3.modelcontext(model)
        if len(vars) != 1:
            raise ValueError("Please provide only one")

        vars = pymc3.inputvars(vars)
        self.__var = vars[0]
        self.__var_name = self.__var.name
        super(BNMetropolis, self).__init__(vars, [model.fastlogp])

    def step(self, point):
        adj = point['network']
        point['network'] = \
            self.__var.distribution.posterior_sample(adj)
        return point
