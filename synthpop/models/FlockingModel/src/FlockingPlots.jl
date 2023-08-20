using Plots

# Utility functions to help with plotting
function angle_of_travel(vel)
    atan(vel[2], vel[1])
end

function rotate_tuple(v::Tuple, angle)
    new_tuple = (cos(angle)*v[1] - v[2]*sin(angle), v[1]*sin(angle) + v[2]*cos(angle))
    return new_tuple
end

# Plotting recipe
@recipe function f(model::BoidModel)
    verts = [(-1,-1), (2,0), (-1, 1)]
    markers = Array{Shape, 1}(undef, model.n)
    
    for i in 1:model.n
        bird_marker = Shape(rotate_tuple.(verts, angle_of_travel(model.vel[i, :])))
        
        markers[i] = bird_marker
    end
    
    # set a default value for an attribute with `-->`
    xlabel --> "x"
    yguide --> "y"
    aspect_ratio := :equal
    xlims := (-250, 250)
    ylims := (-250, 250)
    
    @series begin
        # force an argument with `:=`
        seriestype := :scatter
        # ignore series in legend and color cycling
        primary := false
        # ensure no markers are shown for the error band
        markershape := markers
        markercolor := "black"
        # return series data
        model.pos[:, 1], model.pos[:, 2]
    end
end