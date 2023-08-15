from sbi import inference
from ..abstract import AbstractMetaGenerator

from dataclasses import dataclass, field
import numpy as np


@dataclass
class SMCABC:
    num_particles: int
    num_initial_pop: int
    num_simulations: int
    epsilon_decay: float
    ess_min = None
    kernel_variance_scale: float = 1.0
    use_last_pop_samples: bool = True
    lra: bool = False
    lra_with_weights: bool = False
    sass: bool = False
    sass_fraction: float = 0.25
    sass_expansion_degree: int = 1
    kde: bool = False
    kde_kwargs: dict = field(default_factory=dict)
    kde_sample_weights: bool = False
    return_summary: bool = False

def _generate_fitted_meta_generator(meta_generator, samples):
    class MetaGenerator(AbstractMetaGenerator):
        def __init__(self, samples):
            self.samples = samples

        def __call__(self):
            random_sample = self.samples[np.random.randint(len(self.samples))]
            return meta_generator(random_sample)
    return MetaGenerator(samples)

def smcabc(model, meta_generator, loss, prior, parameters, num_workers=-1):
    """
    Performs SMCABC inference using the SBI pckage.
    """

    def simulator(theta):
        generator = meta_generator(theta)
        x = model(generator)
        return loss(x)

    def distance(y, x):
        return x.reshape(-1)

    simulator, prior = inference.prepare_for_sbi(simulator, prior)
    smcabc_sampler = inference.SMCABC(
        simulator, prior, num_workers=num_workers, distance=distance
    )
    smcabc_sampler.distance = distance  # bug in SBI?
    samples = smcabc_sampler(
        0.0,
        num_particles=parameters.num_particles,
        num_initial_pop=parameters.num_initial_pop,
        num_simulations=parameters.num_simulations,
        epsilon_decay=parameters.epsilon_decay,
        ess_min=parameters.ess_min,
        kernel_variance_scale=parameters.kernel_variance_scale,
        use_last_pop_samples=parameters.use_last_pop_samples,
        lra=parameters.lra,
        lra_with_weights=parameters.lra_with_weights,
        sass=parameters.sass,
        sass_fraction=parameters.sass_fraction,
        sass_expansion_degree=parameters.sass_expansion_degree,
        kde=parameters.kde,
        kde_kwargs=parameters.kde_kwargs,
        kde_sample_weights=parameters.kde_sample_weights,
        return_summary=parameters.return_summary,
    )
    return _generate_fitted_meta_generator(meta_generator, samples)
