# Population synthesis as scenario generation

# 1. Installation

To install the package, clone the repository and run:
```
pip install synthpop
```

# 2. Documentation

You can view the docs [here](https://github.com/joelnmdyer/synthpop/tree/main/notebooks). In particular, you will find examples for how to apply the methods contained in this package to generate populations and scenarios of interest in example agent-based simulators.

# 3. Example

Consider a population of $N$ agents whose states $x_i \sim \mathcal{N}(\mu_i, 1)$, where $\mathcal{N}(\mu, \sigma^2)$ denotes a Normal distribution with mean $\mu$ and variance $\sigma^2$. Consider also generating the agent-level attributes $\mu_i$ from a 
distribution $\iota_\mu = \mathcal{N}(\mu, 1)$. We'd like to find a proposal distribution $q$ over the population-level parameter $\mu$ such that the average square 

$$\ell(\mathbf{x}) = \frac{1}{N} \sum_{i=1}^{N} x_i^2$$

of the agent states is small.

## 3.1 Implementing the model

We implement the model for how agent parameters $\mu_i$ are generated given $\mu$, along with the model for how the agent states $x_i$ are forward simulated given their individual $\mu_i$:

```python
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

        # Specify how the population parameter \mu parameterises the agent generator
        def generator(n_agents):
            # Draw agent parameters from distribution \iota_\mu
            mus = mu + np.random.normal(size=n_agents)
            return mus

        return generator

    def run(self, generator):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Generate agent parameters \mu_i
            mus = generator(self.n_agents)
            # Simulate model forward to obtain the x_i
            xs = mus + np.random.normal(size=self.n_agents)
            return self.observe(xs)
```

## 3.2 Specifying the loss function

We also specify the loss function:

```python
import torch

def loss(x):
    z = torch.mean(torch.pow(x[0], 2))
    return z
```

## 3.3 Wrapping the agent attribute generator

We wrap this for convenience:

```python
class AgentAttributeDistributionGenerator(SampleGenerator):
    def forward(self, generator_params):
        mu = generator_params
        return model.make_default_generator(mu)

meta_generator = AgentAttributeDistributionGenerator()
```

## 3.4 Specify the domain and optimise

Finally, we specify the domain over which we'd like to find such a $q$, and a method for obtaining $q$, before running the optimisation procedure:

```python
prior = torch.distributions.Uniform(torch.tensor([-20.]), torch.tensor([20.]))

optimise = Optimise(model=model, meta_generator=meta_generator, prior=prior, loss=loss)
optimise_method = TBS_SMC(num_particles=5_000, num_initial_pop=10_000, num_simulations=10_000, epsilon_decay=0.7, return_summary=True)
trained_meta_generator = optimise.fit(optimise_method, num_workers=-1)
```
## 3.5 Optimising with variational optimisation

The same example, optimised using variational optimisation, can be seen [here](https://github.com/joelnmdyer/synthpop/blob/main/test/test_normals.py).


# 4. Citation

This package accompanies our [AAMAS 2024](https://www.aamas2024-conference.auckland.ac.nz) paper on [Population synthesis as scenario generation](https://ora.ox.ac.uk/objects/uuid:87663b7f-60ca-44f3-8fa5-b9fd501e6270/download_file?file_format=application%2Fpdf&safe_filename=Dyer_et_al_2023_Population_synthesis_as.pdf&type_of_work=Conference+item) in agent-based models, with the aim of facilitating simulation-based planning under uncertainty. You can cite our paper and/or package using the following:

```
@inproceedings{dyer2023a,
  publisher = {Association for Computing Machinery},
  title = {Population synthesis as scenario generation for simulation-based planning under uncertainty},
  author = {Dyer, J and Quera-Bofarull, A and Bishop, N and Farmer, JD and Calinescu, A and Wooldridge, M},
  year = {2023},
  organizer = {23rd International Conference on Autonomous Agents and Multiagent Systems (AAMAS 2024)},
}
```

The supplementary material (that being this GitHub repository and the [paper appendix](https://github.com/joelnmdyer/synthpop/blob/main/appendix.pdf)) can be cited separately from the main paper as:

```
@software{joel_dyer_2024_10629106,
  author       = {Joel Dyer and
                  Arnau Quera-Bofarull},
  title        = {joelnmdyer/synthpop: AAMAS release},
  month        = feb,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.10629106},
  url          = {https://doi.org/10.5281/zenodo.10629106}
}
```
