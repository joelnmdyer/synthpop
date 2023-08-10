"""
This test is a simple example of how to use the synthpop package.
It is a simple model of a solar system, with the sun at the center
and planets orbiting around it. The sun is stationary, and the planets
move around the sun. The initial conditions determine the type of orbit,
the goal is to generate populations with circular, eliptic, or parabolic orbits.
"""

import torch
import pytest
import torchdiffeq

from synthpop import AbstractModel, AbstractGenerator, AbstractMetaGenerator

M_SUN = 1.989e30
G = 6.674e-11
EARTH_ORBITAL_VELOCITY = 3e4
AU = 1.496e11
DAY = 24 * 60 * 60

class ODE(torch.nn.Module):
    def forward(self, t, x):
        """
        Newton's law with gravitational force
        x here represents [x, y, vx, vy]
        """
        r = torch.norm(x[:, [0,1]], dim=1, keepdim=True)
        a = - G * M_SUN * x[:, [0,1]] / r**3
        return torch.cat([x[:, [2,3]], a], dim=1)

class SolarSystem(AbstractModel):
    def __init__(self, generator, n_agents, t_final, n_timesteps):
        self.generator = generator
        self.n_agents = n_agents
        self.x = None
        self.t_final = t_final * DAY
        self.n_timesteps = n_timesteps
        self.t_range = torch.linspace(0, self.t_final, n_timesteps)

    def initialize(self):
        x = self.generator(self.n_agents)
        # convert x from AU to meters
        x[:, [0,1]] *= AU
        # convert v from orb vel to m/s
        x[:, [2,3]] *= EARTH_ORBITAL_VELOCITY
        return x

    def step(self):
        pass

    def run(self):
        x = self.initialize()
        x = torchdiffeq.odeint(ODE(), x, self.t_range)
        return x

    def observe(self, x):
        return [x[:,:, [0,1]] / AU]


class TestSolarSystem:
    @pytest.fixture(name="ss")
    def make_ss(self):
        generator = lambda n: torch.tensor([1., 2., 3., 4.]) * torch.ones(n, 4)
        ss = SolarSystem(generator, n_agents=2, t_final=60, n_timesteps=10)
        return ss

    def test__initialize(self, ss):
        x = ss.initialize()
        assert x.shape == (2, 4)
        assert x[0,0] == AU
        assert x[0,1] == 2 * AU
        assert x[0,2] == 3 * EARTH_ORBITAL_VELOCITY
        assert x[0,3] == 4 * EARTH_ORBITAL_VELOCITY

    def test__run(self, ss):
        x = ss.run()
        assert x.shape == (10, 2, 4)
        assert x[0,0,0] == AU
        assert x[0,0,1] == 2 * AU
        assert x[0,0,2] == 3 * EARTH_ORBITAL_VELOCITY
        assert x[0,0,3] == 4 * EARTH_ORBITAL_VELOCITY

