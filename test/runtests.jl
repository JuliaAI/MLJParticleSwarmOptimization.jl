using ParticleSwarmOptimization
using Random
using Test
using Distributions
using EvoTrees
using MLJBase
using MLJTuning
using StableRNGs

const PSO = ParticleSwarmOptimization

@testset "Particle Swarm" begin
    include("parameters.jl")
    include("update.jl")
    include("tuning.jl")
    include("problems.jl")
end