import torch
import torchdiffeq

from ..abstract import AbstractModel

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
    def __init__(self, n_agents, t_final, n_timesteps):
        """
        Model representing a solar system where the planets are the agents orbiting the sun.

        **Arguments:**

        - `generator`: a function that generates the initial conditions for the planets.
        The output of the generator should be a tensor of shape (n_agents, 4), where
        the columns represent [x, y, vx, vy] in AU and 3e4 km /s units.
        """
        self.n_agents = n_agents
        self.x = None
        self.t_final = t_final * DAY
        self.n_timesteps = n_timesteps
        self.t_range = torch.linspace(0, self.t_final, n_timesteps)

    def initialize(self, generator):
        x = generator(self.n_agents)
        # convert x from AU to meters
        x[:, [0,1]] *= AU
        # convert v from orb vel to m/s
        x[:, [2,3]] *= EARTH_ORBITAL_VELOCITY
        return x

    def step(self):
        pass

    def observe(self, x):
        return [x[:,:, [0,1]] / AU]

    def run(self, generator):
        x = self.initialize(generator)
        x = self.observe(torchdiffeq.odeint(ODE(), x, self.t_range))
        return x

