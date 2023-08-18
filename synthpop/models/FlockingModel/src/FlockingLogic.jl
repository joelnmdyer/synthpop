using LinearAlgebra
using Statistics
using Plots
using Distributions

# Main logic
struct BoidModel
    n::Int64 # number of birds N
    k::Int64 # number of factors K
    pos::Matrix{Float64} # N by 2 matrix of movement directions
    vel::Matrix{Float64} # N by 2 matrix of bird positions
    speed::Vector{Float64} # N length vector containing speed of each bird
    factors::Matrix{Float64} # K x N  matrix containing factors for each bird
    radii::Matrix{Float64} #  K x N matrix containing radii for each bird and factor pair
end

function BoidModel(n, k, pos, vel, scalar_speed, scalar_factors::Vector{Float64}, scalar_radii::Vector{Float64})
    speed = ones(n) .* scalar_speed
    factors = ones((k, n)) .* scalar_factors
    radii = ones((k, n)) .* scalar_radii
    BoidModel(n, k, pos, vel, speed, factors, radii)
end

function BoidModel(n, k, scalar_speed, scalar_factors::Vector{Float64}, scalar_radii::Vector{Float64})
    # Generate random velocities
    vel = rand(n, 2) .* 2.0 .- 1.0
    vel = vel ./ norm.(eachrow(vel))
    
    # Generate positions evenly space on 50x50 grid
    root = convert(Int64, ceil(sqrt(n)))
    width = collect(range(-50.0, 50, length=root))
    height = collect(range(-50.0, 50, length=root))
    
    pos_grid = ones(Float64, (root * root, 2))
    
    ctr = 1
    for i in 1:root, j in 1:root
        pos_grid[ctr, 1] = width[i]
        pos_grid[ctr, 2] = height[j]
        ctr += 1
    end
    
    pos = pos_grid[1:n, :]

    BoidModel(n, k, pos, vel, scalar_speed, scalar_factors, scalar_radii)
end

# Sample each bird's parameters from some joint distribution/sampler
function BoidModel(n, k, dstr)
    data = rand(dstr, n)
    pos = transpose(data[1:2, :])
    vel = transpose(data[3:4, :])
    speed = data[5, :]
    factor = data[6:6+k-1, :]
    radii = data[6+k:5+2*k, :]

    BoidModel(n, k, pos, vel, speed, factor, radii)
end

function update!(model, factor_fns)
    
    dirs = zeros((model.k, model.n, 2))
    
    # Theres probably a way to do this with broadcasting
    for i in 1:model.k 
        factor_fn = factor_fns[i]
        facts = model.factors[i, :]
        dirs[i, :, :] = factor_fn(model.pos, model.vel, model.radii[i, :]) .* facts
    end

    model.vel[:] = dropdims(sum(dirs, dims=1), dims=1)
    vel_norms = norm.(eachrow(model.vel))
    vel_norms[vel_norms .== 0.0] .= 1.0
    model.vel[:] = model.vel ./ vel_norms
    model.pos[:] = model.pos + model.speed .* model.vel
end

function run!(model, t, factor_fns)
    pos_history = zeros((model.n, 2, t)) # Could make this 
    vel_history = zeros((model.n, 2, t))
    for i in 1:t
        update!(model, factor_fns)
        pos_history[:, :, i] = model.pos
        vel_history[:, :, i] = model.vel
    end
    
    return pos_history, vel_history
end

Base.show(io::IO, model::BoidModel) = print(io, "Boid Model with ", model.n, " birds")
