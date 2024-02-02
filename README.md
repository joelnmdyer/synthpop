# Population synthesis as scenario generation

# 1. Installation

To install the package, clone the repository and run:
```
pip install synthpop
```

# 2. Documentation

You can view the docs [here](https://github.com/joelnmdyer/synthpop/tree/main/notebooks). In particular, you will find examples for how to apply the methods contained in this package to generate populations and scenarios of interest in example agent-based simulators.

# 3. Example

Consider a population of `N` agents whose states $x_i \sim \mathcal{N}(\mu_i, 1)$, where $\mathcal{N}(\mu, \sigma^2)$ denotes a Normal distribution with mean $\mu$ and variance $\sigma^2$. Consider also generating the agent-level attributes $\mu_i$ from a 
distribution $\iota_\mu(\mu_i) = \mathcal{N}(\mu, 1)$. We'd like to find a proposal distribution $q$ over the population-level parameter $\mu$ such that the average square $\ell(\mathbf{x}) = \frac{1}{N} \sum_{i=1}^N x_i^2$ of the agent states is small.

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
