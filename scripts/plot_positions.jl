using Pkg
Pkg.activate("./synthpop/models/FlockingModel")

using FlockingModel
using NPZ, Plots

function plot_positions(positions_path, velocities_path, save_path)
    positions = npzread(positions_path);
    velocities = npzread(velocities_path);
    n_timesteps = size(positions)[3]
    n_agents = size(positions)[1]
    k = 5
    anim = @animate for i in 1:n_timesteps
        bird_pos = positions[:, :, i]
        bird_vel = velocities[:, :, i]
        model = BoidModel(n_agents, k, bird_pos, bird_vel, zeros(1), zeros(1), zeros(1))
        plot(model)
    end
    gif(anim, "first_example.gif", fps = 24)
end
