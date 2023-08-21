import torch
import normflows as nf

class SampleGenerator(torch.nn.Module):
    def forward(self, generator_params):
        raise NotImplementedError

class DiracDelta(torch.nn.Module):
    def __init__(self, parameter_priors):
        super().__init__()
        self.__parameters = torch.nn.Parameter(parameter_priors)

    def sample(self, n_samples):
        samples = self.__parameters.repeat(n_samples, 1)
        return samples, self.log_prob(samples)

    def log_prob(self, x):
        # approximate by very thing gaussian
        return torch.distributions.Normal(self.__parameters, 1e-2 * torch.ones_like(self.__parameters)).log_prob(x).sum(-1)

    def forward(self, generator_params):
        raise NotImplementedError
    


class MultivariateNormal(torch.nn.Module):
    def __init__(self, mean, sqrt_cov):
        super().__init__()
        self.mean = mean
        self.sqrt_cov = sqrt_cov
        self.device = mean.device
        self._params = torch.nn.ParameterList([self.mean, self.sqrt_cov])

    def sample(self, n_samples):
        cov = torch.mm(self.sqrt_cov, self.sqrt_cov.t())
        dist = torch.distributions.MultivariateNormal(self.mean, cov)
        samples = dist.rsample((n_samples,))
        return samples, dist.log_prob(samples)

    def log_prob(self, x):
        cov = torch.mm(self.sqrt_cov, self.sqrt_cov.t())
        dist = torch.distributions.MultivariateNormal(self.mean, cov)
        return dist.log_prob(x)

    def parameters(self):
        return [self.mean, self.sqrt_cov]

    def forward(self, generator_params):
        raise NotImplementedError

class MaskedAutoRegressiveFlow(torch.nn.Module):
    def __init__(self, n_parameters, n_hidden_units, n_transforms):
        super().__init__()
        self.n_parameters = n_parameters
        self.n_hidden_units = n_hidden_units
        self.n_transforms = n_transforms
        self.flow = self._build()

    def _build(self):
        flows = []
        for i in range(self.n_transforms):
            flows.append(
                nf.flows.MaskedAffineAutoregressive(self.n_parameters, self.n_hidden_units, num_blocks=2)
            )
            flows.append(nf.flows.LULinearPermute(self.n_parameters))
        q0 = nf.distributions.DiagGaussian(self.n_parameters, trainable=False)
        return nf.NormalizingFlow(q0=q0, flows=flows)

    def sample(self, n_samples):
        return self.flow.sample(n_samples)

    def log_prob(self, x):
        return self.flow.log_prob(x)

    def parameters(self):
        return self.flow.parameters()

    def forward(self, generator_params):
        raise NotImplementedError


