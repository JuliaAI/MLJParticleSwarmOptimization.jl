module ParticleSwarmOptimization

using Random
using Distributions
using MLJBase
using MLJTuning

export ParticleSwarm

include("types.jl")
include("parameters.jl")
include("move.jl")
include("update.jl")

end