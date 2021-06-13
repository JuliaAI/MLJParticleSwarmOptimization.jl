module ParticleSwarmOptimization

using Random
using Distributions
using MLJBase
using MLJTuning
using StatsBase

export ParticleSwarm

include("types.jl")
include("initialize.jl")
include("update.jl")

end