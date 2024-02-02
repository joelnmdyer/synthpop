import numpy as np
import warnings
from ..abstract import AbstractModel

class Normals(AbstractModel):
    def __init__(self, n_timesteps=1, n_agents=1_000):
        self.n_timesteps = n_timesteps
        self.n_agents = n_agents

    def initialize(self):
        pass

    def step(self, *args, **kwargs):
        pass

    def observe(self, x):
        return [x]

    @staticmethod
    def make_default_generator(params):
        mu = params

        def generator(n_agents):
            # Draw agent parameters
            mus = mu + np.random.normal(size=n_agents)
            return mus

        return generator

    def run(self, generator):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mus = generator(self.n_agents)
            # Simulate model forward
            xs = mus + np.random.normal(size=self.n_agents)
            return self.observe(xs)
