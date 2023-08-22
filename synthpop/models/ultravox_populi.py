from numba import njit
import warnings
import networkx as nx
import numpy as np
from tqdm import tqdm
from random import random

from ..abstract import AbstractModel


@njit
def random_choice_numba(arr, prob):
    return arr[np.searchsorted(np.cumsum(prob), random(), side="right")]


@njit
def _step(os, mus, gammas, N, neighbours):
    new_os = os.copy()
    for i in range(N):
        is_neighbours = neighbours[i]
        average_opinion = os[is_neighbours].mean()
        if os[i] == 0:
            prob_1_given_0 = average_opinion * mus[i]
            prob = prob_1_given_0
            # prob = 1 + mus[i] * (average_opinion - 1)
            # print(i, 0, prob)
            if np.random.random() < prob:
                new_os[i] = 1
            else:
                new_os[i] = 0
            # new_os[i] = (np.random.random() < prob).astype(int)
        else:
            prob_0_given_1 = (1 - average_opinion) * gammas[i]
            prob = 1 - prob_0_given_1
            # prob = gammas[i] * average_opinion
            # print(i, 1, prob)
            if np.random.random() < prob:
                new_os[i] = 1
            else:
                new_os[i] = 0
            # new_os[i] = (np.random.random() < prob).astype(int)
    return new_os


@njit
def _simulate(os, mus, gammas, each_agents_neighbours, T, N):
    opinions = []

    for t in range(T):
        os = _step(os, mus, gammas, N, each_agents_neighbours)
        opinions.append(os)

    return opinions


class UltravoxPopuli(AbstractModel):
    def __init__(self, n_timesteps=100_000, n_agents=1_000):
        self.n_timesteps = n_timesteps
        self.n_agents = n_agents

    def initialize(self):
        pass

    def step(self, *args, **kwargs):
        pass

    def observe(self, x):
        return [np.array([np.mean(o) for o in x])]

    @staticmethod
    def make_default_generator(params):
        r, alpha_mu, beta_mu, alpha_gamma, beta_gamma = params

        def generator(n_agents):
            # Draw initial opinions
            #os = np.random.beta(alpha_os, beta_os, size=n_agents)
            os = np.random.binomial(1, r, n_agents)
            #os = (np.random.random(size=n_agents) < r).astype(int)
            # Draw malleabilities
            mus = np.random.beta(alpha_mu, beta_mu, size=n_agents)
            gams = np.random.beta(alpha_gamma, beta_gamma, size=n_agents)
            return os, mus, gams

        return generator

    def run(self, generator):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            os, mus, gams = generator(self.n_agents)

            graph = nx.generators.barabasi_albert_graph(self.n_agents, 2)
            each_agents_neighbours = [
                np.array([nbr for nbr in graph.neighbors(i)]) for i in range(self.n_agents)
            ]
            x = [os.copy()] + _simulate(
                os, mus, gams, each_agents_neighbours, self.n_timesteps, self.n_agents
            )
            return self.observe(x)

    def reconstruct_opinions(self, last_opinions, agent, opinion):
        last_opinions[agent] = opinion
        return last_opinions
