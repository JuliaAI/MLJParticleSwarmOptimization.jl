module MLJParticleSwarmOptimization

using LinearAlgebra
using Random
using Distributions
using MLJBase
using MLJTuning

export ParticleSwarm, AdaptiveParticleSwarm

include("interface.jl")
include("parameters.jl")
include("update.jl")
include("strategies/basic.jl")
include("strategies/adaptive.jl")

end