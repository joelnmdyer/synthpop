import torch
import pytest

from synthpop.models.solar_system import SolarSystem, AU, EARTH_ORBITAL_VELOCITY

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