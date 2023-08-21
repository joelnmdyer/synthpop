from julia import Julia
import numpy as np
import logging
import os
from pathlib import Path
from synthpop.abstract import AbstractModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FlockingModel")

logger.info("Loading julia...")
jl = Julia(compiled_modules=False)

from julia import Main

env_path = Path(os.path.abspath(__file__)).parent / "FlockingModel"
Main.eval("using Pkg")
Main.eval(f"Pkg.activate(\"{env_path}\")")


run_script_jl = """using FlockingModel, Plots
function run_flocking_model(n, k, pos, vel, speed, factors, radii; time_steps)
    factor_fns = [avoid_direction, stubborn_direction, sep_direction, cohere_direction, match_direction]
    model = BoidModel(n, k, pos, vel, speed, factors, radii)
    run!(model, time_steps, factor_fns)
end
function plot_positions(positions, velocities, save_path, plot_lim)
    FlockingModel.PLOT_LIM = plot_lim
    n_timesteps = size(positions)[3]
    n_agents = size(positions)[1]
    k = 5
    anim = @animate for i in 1:n_timesteps
        bird_pos = positions[:, :, i]
        bird_vel = velocities[:, :, i]
        model = BoidModel(n_agents, k, bird_pos, bird_vel, zeros(1), zeros(1), zeros(1))
        plot(model)
    end
    gif(anim, save_path, fps = 24)
end
"""
logger.info("Pre-compiling FlockingModel...")
Main.eval(run_script_jl)


class FlockingModel(AbstractModel):
    def __init__(self, n_timesteps, n_agents, k):
        self.n_timesteps = n_timesteps
        self.n_agents = n_agents
        self.k = k

    def initialize(self, generator):
        pass

    def step(self):
        pass

    def observe(self, x):
        return x

    def run(self, generator):
        pos, vel, speed, factors, radii = generator(self.n_agents)
        pos_hist, vel_hist = Main.run_flocking_model(
            self.n_agents,
            self.k,
            pos,
            vel,
            speed,
            factors,
            radii,
            time_steps=self.n_timesteps,
        )
        return [pos_hist, vel_hist]

    def plot(self, positions, velocities, save_path = "birds.gif", plot_lim=500):
        return Main.plot_positions(positions, velocities, save_path, plot_lim)


if __name__ == "__main__":
    import numpy as np
    n = 5
    k = 3
    pos = np.random.rand(n, 2)
    vel = np.random.rand(n, 2)
    speed = np.random.rand(n)
    factors = np.random.rand(k, n)
    radii = np.random.rand(k,n)

    fm = FlockingModel(10, n, k)
    results = fm.run(lambda n: (pos, vel, speed, factors, radii))