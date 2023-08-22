import torch
import numpy as np
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

#class Sigmoid(nf.flows.Flow):
#    def __init__(self, min_values, max_values):
#        super().__init__()
#        self.min_values = min_values
#        self.max_values = max_values
#
#    def forward(self, z):
#        sim = torch.sigmoid(z)
#        zz = self.min_values + (self.max_values - self.min_values) * sim
#        log_det = sum(np.abs(b-a) for (a, b) in zip(self.min_values, self.max_values)) 
#        log_det += torch.sum(torch.log(torch.abs(torch.exp(-z) / (1 + torch.exp(-z))**2)))
#        return zz, log_det
#
#    def inverse(self, z):
#        x = torch.log((z - self.min_values) / (self.max_values - z))
#        log_det = torch.sum(torch.log(torch.abs((self.max_values - self.min_values) / (self.max_values - z) * (z / (z - self.min_values)))))
#        return x, log_det
class Sigmoid(nf.flows.Flow):
    def __init__(self, min_values, max_values):
        super().__init__()
        self.min_values = min_values
        self.max_values = max_values


    def inverse(self, z):
        logz = torch.log(z - self.min_values)
        log1mz = torch.log(self.max_values - z)
        z = logz - log1mz
        sum_dims = list(range(1, z.dim()))
        log_det = (
            - torch.sum(logz, dim=sum_dims)
            - torch.sum(log1mz, dim=sum_dims)
        )
        return z, log_det

    def forward(self, z):
        sum_dims = list(range(1, z.dim()))
        ls = torch.sum(torch.nn.functional.logsigmoid(z), dim=sum_dims)
        mls = torch.sum(torch.nn.functional.logsigmoid(-z), dim=sum_dims)
        lls = torch.sum(torch.log(self.max_values - self.min_values))
        log_det = ls + mls + lls
        z = self.min_values + (self.max_values - self.min_values) * torch.sigmoid(z)
        return z, log_det

class MaskedAutoRegressiveFlow(torch.nn.Module):
    def __init__(self, n_parameters, n_hidden_units, n_transforms, min_values, max_values):
        super().__init__()
        self.n_parameters = n_parameters
        self.n_hidden_units = n_hidden_units
        self.n_transforms = n_transforms
        self.min_values = min_values
        self.max_values = max_values
        self.flow = self._build()

    def _build(self):
        flows = []
        for i in range(self.n_transforms):
            flows.append(
                nf.flows.MaskedAffineAutoregressive(self.n_parameters, self.n_hidden_units, num_blocks=2)
            )
            flows.append(nf.flows.LULinearPermute(self.n_parameters))
        flows.append(Sigmoid(min_values = self.min_values, max_values = self.max_values))
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


class NeuralSplineFlow(torch.nn.Module):
    def __init__(self, n_parameters, n_hidden_units, n_hidden_layers, n_transforms, min_values, max_values):
        super().__init__()
        self.n_parameters = n_parameters
        self.n_hidden_units = n_hidden_units
        self.n_hidden_layers = n_hidden_layers
        self.n_transforms = n_transforms
        self.min_values = min_values
        self.max_values = max_values
        self.flow = self._build()

    def _build(self):
        flows = []
        for i in range(self.n_transforms):
            flows += [nf.flows.AutoregressiveRationalQuadraticSpline(self.n_parameters, self.n_hidden_layers, self.n_hidden_units)]
            flows += [nf.flows.LULinearPermute(self.n_parameters)]

        flows.append(Sigmoid(min_values = self.min_values, max_values = self.max_values))
        # Set base distribuiton
        q0 = nf.distributions.DiagGaussian(self.n_parameters, trainable=False)

        # Construct flow model
        flow = nf.NormalizingFlow(q0=q0, flows=flows)
        return flow

    def sample(self, n_samples):
        return self.flow.sample(n_samples)

    def log_prob(self, x):
        return self.flow.log_prob(x)

    def parameters(self):
        return self.flow.parameters()

    def forward(self, generator_params):
        raise NotImplementedError



