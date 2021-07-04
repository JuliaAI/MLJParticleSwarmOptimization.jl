module ParticleSwarmOptimization

using Random
using Distributions
using MLJBase
using MLJTuning

export StaticCoeffs, ParticleSwarm

include("swarm.jl")
include("parameters.jl")
include("update.jl")
include("tuning.jl")

end