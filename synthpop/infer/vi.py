from dataclasses import dataclass
from blackbirds import infer as bb_infer
import numpy as np
import torch

@dataclass
class VI:
    w: float
    optimizer: torch.optim.Optimizer
    gradient_clipping_norm: float = np.inf
    n_samples_per_epoch: int = 10
    n_samples_regularisation: int = 10_000
    diff_mode: str = "reverse"
    initialize_estimator_to_prior: bool = False
    gradient_estimation_method: str = "pathwise"
    jacobian_chunk_size: int | None = None
    progress_bar: bool = False
    progress_info: bool = True
    log_tensorboard: bool = False
    tensorboard_log_dir: str | None = None

def vi(model, meta_generator, loss, prior, parameters, n_epochs, max_epochs_without_improvement = np.inf):
    def _loss(params, _):
        generator = meta_generator(params)
        x = model(generator)
        return loss(x)

    vi = bb_infer.VI(
        loss=_loss,
        posterior_estimator=meta_generator,
        prior=prior,
        optimizer=parameters.optimizer,
        n_samples_per_epoch=parameters.n_samples_per_epoch,
        w=parameters.w,
        log_tensorboard=parameters.log_tensorboard,
        gradient_estimation_method=parameters.gradient_estimation_method,
        gradient_clipping_norm=parameters.gradient_clipping_norm,
        diff_mode=parameters.diff_mode,
        initialize_estimator_to_prior=parameters.initialize_estimator_to_prior,
        device="cpu",
        jacobian_chunk_size=parameters.jacobian_chunk_size,
        progress_bar=parameters.progress_bar,
        progress_info=parameters.progress_info,
        tensorboard_log_dir=parameters.tensorboard_log_dir,
    )
    vi.run(None, n_epochs=n_epochs, max_epochs_without_improvement=max_epochs_without_improvement)
    return meta_generator


