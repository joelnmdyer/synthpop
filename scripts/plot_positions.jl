using FlockingModel

using NPZ, Plots


##
positions = npzread("./notebooks/positions.npy");
velocities = npzread("./notebooks/velocities.npy");

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
##