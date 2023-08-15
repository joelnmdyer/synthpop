module FlockingModel

include("FactorFunctions.jl")
include("FlockingLogic.jl")
include("FlockingPlots.jl")

export BoidModel, run!, update!

export cohere_direction, match_direction, avoid_direction, stubborn_direction, sep_direction

end # module FlockingModel