using Distances

stubborn_direction(vel) = vel

stubborn_direction(pos, vel, radii) = stubborn_direction(vel)

function get_mask(pos, radii)
    # norms = [norm(x - y) for x in eachrow(pos), y in eachrow(pos)]
    norms = pairwise(Euclidean(), pos, pos, dims=1)
    mask = norms .< radii .&& norms .> 0.0
    mask = convert(Matrix{Float64}, mask)
    return mask
end

function cohere_direction(pos, radii)
    mask = get_mask(pos, radii)
    # Prevent division by zero
    nbr_cnt = sum(mask, dims=2)
    nbr_cnt = max.(nbr_cnt, 1)
    nbr_sums= mask * pos
    nbr_ctr = nbr_sums ./ nbr_cnt
    nbr_direction =  nbr_ctr .-  pos
    # Theres probably a way to write this with circuit breakers which is one line
    nbr_norms = norm.(eachrow(nbr_direction))
    nbr_norms[nbr_norms .== 0.0] .= 1.0
    nbr_direction = nbr_direction ./ nbr_norms
    return nbr_direction
end

cohere_direction(pos, vel, radii) = cohere_direction(pos, radii)

function sep_direction(pos, radii)
    mask = get_mask(pos, radii)
    n = size(pos)[1]
    hds = [pos[j, k] - pos[i, k]  for i in 1:n, j in 1:n, k in 1:2] # headings
    # hd_norms = [norm(hds[i, j, :]) for i in 1:n, j in 1:n] # heading norms
    hd_norms = reshape(norm.(eachrow(reshape(hds, :, 2))), (size(hds, 1), size(hds, 1)))
    hd_norms[hd_norms .== 0.0] .= 1.0
    nhds = hds ./ hd_norms # normalised headings
    mhds = mask .* nhds # masked out headings
    sep = dropdims(sum(mhds, dims=2), dims=2) # so ugly...
    sep_norms = norm.(eachrow(sep))
    sep_norms[sep_norms .== 0.0] .= 1.0
    sep = sep ./ sep_norms
    return -sep
end

sep_direction(pos, vel, radii) = sep_direction(pos, radii)



function match_direction(pos, vel, radii)
    mask = get_mask(pos, radii)
    vel_norms = norm.(eachrow(vel))
    vel_norms[vel_norms .== 0.0] .= 1.0
    nvel = vel ./ vel_norms
    mat = mask * nvel
    mat_norms = norm.(eachrow(mat))
    mat_norms[mat_norms .== 0.0] .= 1.0
    mat = mat ./ mat_norms
    return mat
end

function avoid_wall(pos, radii, idx, wall, dir)
    n = size(pos)[1]
    loc = pos[:, idx]
    mask = abs.(loc .- wall) .< radii .|| dir .* wall .- loc .> 0
    ret_dirs = zeros(Float64, (n, 2))
    ret_dirs[:, idx] = dir .* convert.(Float64, mask)
    return ret_dirs
end

function avoid_direction(pos, radii)
    dirs_1 = avoid_wall(pos, radii, 1, -250.0, 1.0)
    dirs_2 = avoid_wall(pos, radii, 1, 250.0, -1.0)
    dirs_3 = avoid_wall(pos, radii, 2, -250.0, 1.0)
    dirs_4 = avoid_wall(pos, radii, 2, 250.0, -1.0)
    
    dirs = dirs_1 + dirs_2 + dirs_3 + dirs_4
    norms = norm.(eachrow(dirs))
    norms[norms .== 0.0] .= 1.0
    dirs = dirs ./ norms
    return dirs
end

avoid_direction(pos, vel, radii) = avoid_direction(pos, radii)