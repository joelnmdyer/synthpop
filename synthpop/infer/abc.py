from typing import Callable
from tqdm import tqdm

from ..abstract import AbstractModel, AbstractGenerator, AbstractMetaGenerator

class IntervalKernel:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def __call__(self, loss, x):
        return loss(x) < self.epsilon

class RejectionSampling:
    def __init__(
        self,
        model: AbstractModel,
        meta_generator: AbstractMetaGenerator,
        loss: Callable,
        kernel: Callable,
        proposal_distribution: Callable,
    ):
        """
        Trains the meta-generator to generate generators that generate initial conditions for the model.
        """
        self.model = model
        self.meta_generator = meta_generator
        self.loss = loss
        self.kernel = kernel
        self.proposal_distribution = proposal_distribution

    def sample(self, n_tries):
        ret = []
        for i in tqdm(range(n_tries)):
            psample = self.proposal_distribution.sample()
            gsample = self.meta_generator(psample)
            x = self.model(gsample)
            accept = self.kernel(self.loss, x)
            if accept:
                ret.append(psample)
        return ret


