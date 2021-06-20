using ParticleSwarmOptimization
using Distributions
using MLJBase
using StableRNGs
using Test

const PSO = ParticleSwarmOptimization

include("parameters.jl")
include("move.jl")
include("update.jl")