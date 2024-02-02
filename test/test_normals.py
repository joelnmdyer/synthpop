import numpy as np
from scipy.stats import ncx2
import torch

from synthpop.generative import SampleGenerator
from synthpop.models import normals
from synthpop.optimise import Optimise, TBS_SMC, VO

class TestNormals:

    def test_tbs_smc(self):
        torch.manual_seed(0)
        n_agents = 1_000
        model = normals.Normals(n_timesteps=1, n_agents=n_agents)

        def loss(x):
            z = torch.mean(torch.pow(x[0], 2))
            return z

        prior = torch.distributions.Uniform(torch.tensor([-20.]), torch.tensor([20.]))

        class AgentAttributeDistributionGenerator(SampleGenerator):
            def forward(self, generator_params):
                mu = generator_params
                return model.make_default_generator(mu)

        meta_generator = AgentAttributeDistributionGenerator()
        optimise = Optimise(model=model, meta_generator=meta_generator, prior=prior, loss=loss)
        optimise_method = TBS_SMC(num_particles=5_000, num_initial_pop=10_000, num_simulations=10_000, epsilon_decay=0.7, return_summary=True)
        trained_meta_generator = optimise.fit(optimise_method, num_workers=-1)
        samples = trained_meta_generator.samples
        fepsilon = trained_meta_generator.final_epsilon
        mean = samples.mean()
        assert np.isclose(mean, 0., atol=2e-2)
        # TODO: check that the learned proposal distribution approximately matches a distribution proportional to CDF of non-central chi-squared at n_agents * fepsilon / 2 

    def test_vo_normal(self):
        torch.manual_seed(0)
        model = normals.Normals(n_timesteps=1, n_agents=1_000)

        b = 10.
        def loss(x):
            x = x[0]
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            return torch.mean(torch.pow(x, 2)) - b

        class NormalMetaGenerator(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.mu0 = torch.nn.Parameter(torch.tensor(5.))

            def forward(self, generator_params=None):
                if generator_params is None:
                    generator_params = self.sample(1)[0][0]
                return model.make_default_generator(generator_params)

            def sample(self, n_samples, *args, **kwargs):
                _normal = torch.distributions.normal.Normal(loc=self.mu0, scale=1.)
                samples = _normal.rsample((n_samples,), *args, **kwargs)
                return samples, _normal.log_prob(samples)

            def log_prob(self, *args, **kwargs):
                _normal = torch.distributions.normal.Normal(loc=self.mu0, scale=1.)
                return _normal.log_prob(*args, **kwargs)

            def parameters(self):
                return [self.mu0]

        normal_meta_generator = NormalMetaGenerator()
        dom_vo = torch.distributions.Uniform(torch.tensor([-20.]), torch.tensor([20.]))
        optimise = Optimise(model=model, meta_generator=normal_meta_generator, prior=dom_vo, loss=loss)
        optimizer = torch.optim.AdamW(normal_meta_generator.parameters(), lr=5e-3)
        gamma = 0.01
        optimise_method = VO(w=gamma, 
                            n_samples_per_epoch=100, 
                            optimizer=optimizer, 
                            progress_bar=True, 
                            progress_info=True, 
                            gradient_estimation_method="score", 
                            log_tensorboard=True, 
                            gradient_clipping_norm=10.0,
                            )
        optimise.fit(optimise_method, n_epochs=2000, max_epochs_without_improvement=50)
        samples, log_probs = normal_meta_generator.sample(1000)
        # Run model forward at each of the samples, evaluate loss
        losses = [loss(model.run(model.make_default_generator(samples[i].detach().numpy()))) for i in range(samples.shape[0])]
        log_probs = normal_meta_generator.log_prob(samples).detach().numpy()
        assert np.isclose(np.mean(losses) + np.mean(log_probs), 3 - 0.5*gamma*np.log(2*np.pi*np.exp(1.)) - b, atol=1.)
        assert np.isclose(normal_meta_generator.mu0.detach().numpy(), 0., atol=1.)
