import logging

import numpy as np
import scipy as sp
import pandas as pd
import pymc3 as pm
import theano.tensor as tt
from sklearn.preprocessing import LabelEncoder

from shm.distributions.binary_mrf import BinaryMRF
from shm.distributions.categorical_mrf import CategoricalMRF
from shm.family import Family
from shm.globals import READOUT, GENE, CONDITION, INTERVENTION
from shm.link import Link
from shm.models.shm import SHM

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


class SHLM(SHM):
    def __init__(self,
                 data: pd.DataFrame,
                 family="gaussian",
                 link_function=Link.identity,
                 model="clustering",
                 n_states=2,
                 graph=None,
                 sampler="nuts"):
        self._data = data
        self._data = self._data.sort_values([GENE, CONDITION, INTERVENTION])

        self._set_link(link_function)
        self._set_family(family)
        self._set_data()

        super().__init__(model=model,
                         n_states=n_states,
                         graph=graph,
                         sampler=sampler)

        if graph:
            d_genes = sp.sort(sp.unique(self._data.gene.values))
            if not sp.array_equal(d_genes, self.node_labels):
                raise ValueError("Graph nodes != data genes")

    @property
    def tau_g_alpha(self):
        return 5

    @property
    def tau_b_alpha(self):
        return 3

    @property
    def tau_iota_alpha(self):
        return 3

    @property
    def sd_alpha(self):
        return 2

    @property
    def edge_correction(self):
        return .5

    @property
    def gamma_means(self):
        if self.n_states == 2:
            return np.array([0., 0.])
        return np.array([-1, 0., 1.])

    @property
    def data(self):
        return self._data

    @property
    def family(self):
        return self._family

    @property
    def link(self):
        return self._link_function

    def _set_link(self, link_function):
        if isinstance(link_function, str):
            link_function = Link.from_str(link_function)
        self._link_function = link_function

    def _set_family(self, family):
        if isinstance(family, str):
            family = Family.from_str(family)
        self._family = family

    def _set_mrf_model(self):
        with pm.Model() as model:
            if self.n_states == 2:
                logger.info("Using binary-MRF")
                z = BinaryMRF('z', G=self.graph, beta=self.edge_correction)
            else:
                logger.info("Using categorical-MRF with three states")
                z = CategoricalMRF('z', G=self.graph, k=3)
        tau_g, mean_g, gamma = self._gamma_mix(model, z)
        param_hlm = self._hlm(model, gamma)

        self._set_steps(model, z, tau_g, mean_g, gamma, *param_hlm)
        return self

    def _set_clustering_model(self):
        with pm.Model() as model:
            logger.info("Using {} cluster centers".format(self.n_states))
            p = pm.Dirichlet(
              "p", a=np.repeat(1, self.n_states), shape=self.n_states)
            pm.Potential("p_pot", var=tt.switch(tt.min(p) < 0.05, -np.inf, 0.))
            z = pm.Categorical("z", p=p, shape=self.n_genes)
        tau_g, mean_g, gamma = self._gamma_mix(model, z)
        param_hlm = self._hlm(model, gamma)

        self._set_steps(model, z, p, tau_g, mean_g, gamma, *param_hlm)
        return self

    def _set_simple_model(self):
        with pm.Model() as model:
            logger.info("Using tau_g_alpha: {}".format(self.tau_g_alpha))
            tau_g = pm.InverseGamma(
              "tau_g", alpha=self.tau_g_alpha, beta=1., shape=1)
            mean_g = pm.Normal("mu_g", mu=0, sd=1, shape=1)
            gamma = pm.Normal("gamma", mean_g, tau_g, shape=self.n_genes)
        param_hlm = self._hlm(model, gamma)

        self._set_steps(model, None, tau_g, mean_g, gamma, *param_hlm)
        return self

    def _gamma_mix(self, model, z):
        with model:
            logger.info("Using tau_g_alpha: {}".format(self.tau_g_alpha))
            tau_g = pm.InverseGamma(
              "tau_g", alpha=self.tau_g_alpha, beta=1., shape=self.n_states)

            logger.info("Using mean_g: {}".format(self.gamma_means))
            if self.n_states == 2:
                logger.info("Building two-state model")
                mean_g = pm.Normal(
                  "mu_g", mu=self.gamma_means, sd=1, shape=self.n_states)
                pm.Potential(
                  "m_opot",
                  var=tt.switch(mean_g[1] - mean_g[0] < 0., -np.inf, 0.))
            else:
                logger.info("Building three-state model")
                mean_g = pm.Normal(
                  "mu_g", mu=self.gamma_means, sd=1, shape=self.n_states)
                pm.Potential(
                  'm_opot',
                  tt.switch(mean_g[1] - mean_g[0] < 0, -np.inf, 0)
                  + tt.switch(mean_g[2] - mean_g[1] < 0, -np.inf, 0))

            gamma = pm.Normal("gamma", mean_g[z], tau_g[z], shape=self.n_genes)

        return tau_g, mean_g, gamma

    def _hlm(self, model, gamma):
        with model:
            logger.info("Using tau_b_alpha: {}".format(self.tau_b_alpha))
            tau_b = pm.InverseGamma(
              "tau_b", alpha=self.tau_b_alpha, beta=1., shape=1)
            beta = pm.Normal("beta", 0, sd=tau_b, shape=self.n_gene_condition)

            logger.info("Using tau_iota_alpha: {}".format(self.tau_iota_alpha))
            l_tau = pm.InverseGamma(
              "tau_iota", alpha=self.tau_iota_alpha, beta=1., shape=1)
            l = pm.Normal("iota", mu=0, sd=l_tau, shape=self.n_interventions)

            mu = (gamma[self._gene_data_idx] +
                  beta[self._gene_cond_data_idx] +
                  l[self._intervention_data_idx])

            if self.family == Family.gaussian:
                logger.info("Using sd_alpha: {}".format(self.sd_alpha))
                sd = pm.InverseGamma("sd", alpha=self.sd_alpha, beta=1., shape=1)
                pm.Normal("x",
                          mu=mu,
                          sd=sd,
                          observed=np.squeeze(self.data[READOUT].values))
            else:
                raise NotImplementedError("Only gaussian family so far")

        return tau_b, beta, l_tau, l, sd

    @property
    def n_genes(self):
        return self.__len_genes

    @property
    def n_conditions(self):
        return self.__len_conds

    @property
    def n_interventions(self):
        return self.__len_intrs

    @property
    def _intervention_data_idx(self):
        return self.__intrs_data_idx

    @property
    def _gene_cond_data_idx(self):
        return self.__gene_cond_data_idx

    @property
    def _index_to_gene(self):
        return self.__index_to_gene

    @property
    def _gene_to_index(self):
        return self.__gene_to_index

    @property
    def _index_to_condition(self):
        return self.__index_to_con

    @property
    def _beta_index_to_gene(self):
        return self.__beta_idx_to_gene

    @property
    def _gene_to_beta_index(self):
        return self.__gene_to_beta_idx

    @property
    def _beta_idx_to_gene_cond(self):
        return self.__beta_idx_to_gene_cond

    @property
    def _gene_data_idx(self):
        return self.__gene_data_idx

    @property
    def n_gene_condition(self):
        return self.__len_gene_cond

    @property
    def _beta_idx(self):
        return self.__beta_idx

    def _set_data(self):
        data = self._data
        self._n, _ = data.shape
        le = LabelEncoder()

        self.__gene_data_idx = le.fit_transform(data[GENE].values)
        self.__index_to_gene = {i: e for i, e in zip(
          self.__gene_data_idx, data[GENE].values)}
        self.__gene_to_index = {e: i for i, e in zip(
          self.__gene_data_idx, data[GENE].values)}
        self.__genes = sp.unique(list(self.__index_to_gene.values()))
        self.__len_genes = len(self.__genes)

        self.__con_data_idx = le.fit_transform(data[CONDITION].values)
        self.__index_to_con = {i: e for i, e in zip(
          self.__con_data_idx, data[CONDITION].values)}
        self.__conditions = sp.unique(list(self.__index_to_con.values()))
        self.__len_conds = len(self.__conditions)

        self.__intrs_data_idx = le.fit_transform(data[INTERVENTION].values)
        self.__index_to_intervention = {i: e for i, e in zip(
          self.__intrs_data_idx, data[INTERVENTION].values)}
        self.__intrs = sp.unique(data[INTERVENTION].values)
        self.__len_intrs = len(self.__intrs)

        self.__beta_idx = sp.repeat(sp.unique(self.__gene_data_idx),
                                    len(self.__conditions))
        self.__beta_idx_to_gene = {i: self.__index_to_gene[i]
                                   for i in self.__beta_idx}
        self.__gene_to_beta_idx = {e: i for i, e in self.__beta_idx_to_gene.items()}

        self.__gene_cond_data = ["{}-{}".format(g, c)
           for g, c in zip(data[GENE].values, data[CONDITION].values)]
        self.__gene_cond_data_idx = le.fit_transform(self.__gene_cond_data)
        self.__len_gene_cond = len(sp.unique(self.__gene_cond_data))
        self.__beta_idx_to_gene_cond = {
            i: e for i, e in zip(self.__gene_cond_data_idx,
                                 self.__gene_cond_data)
        }

