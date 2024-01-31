import torch
from typing import Callable
from dataclasses import dataclass

from ..abstract import AbstractModel, AbstractMetaGenerator

from .tbs_smc import tbs_smc as tbs_smc
from .vi import vi as vo


class Optimise:
    def __init__(
        self,
        model: AbstractModel,
        meta_generator: AbstractMetaGenerator,
        loss: Callable,
        prior: torch.distributions.Distribution,
    ):
        """
        General class to optimise over the MetaGenerator parameters.

        **Arguments:**

        - `model`: Simulation model, must take a structural parameter vector omega, and output a list of outputs that
        is compatible with the input of the loss function.
        - `meta_generator`: Object that generates attribute distributions for the simulator. Must be callable by
        passing a set of population-level parameters theta and outputing a distribution that can be taken by the model as input.
        - `loss`: Loss function that takes the output of the model and the target data and outputs a scalar, representing
        how well the model has generated the desired scenario.
        """
        self.model = model
        self.meta_generator = meta_generator
        self.loss = loss
        self.prior = prior

    def fit(self, method: dataclass, **kwargs):
        """
        Fit the model to the data using the given method.

        **Arguments:**

        - `method`: Method to use for fitting. Currently only 'MLE' is supported.
        - `num_workers`: Number of workers to use for parallelization. If -1, the number of workers will be set to the
        number of available CPUs.
        """
        method_name = method.__class__.__name__
        if method_name == "TBS-SMC":
            return tbs_smc(
                model=self.model,
                meta_generator=self.meta_generator,
                loss=self.loss,
                prior=self.prior,
                parameters=method,
                **kwargs
            )
        elif method_name == "VO":
            return vo(
                model=self.model,
                meta_generator=self.meta_generator,
                loss=self.loss,
                prior=self.prior,
                parameters=method,
                **kwargs
            )
        else:
            raise NotImplementedError(f"Method {method_name} not implemented.")
