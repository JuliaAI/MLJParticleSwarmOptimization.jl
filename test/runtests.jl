using ParticleSwarmOptimization
using Distributions
using MLJBase
using Random
using StableRNGs
using Test

const PSO = ParticleSwarmOptimization

@testset "Particle Swarm" begin
    include("options.jl")
    include("swarm.jl")
    include("parameters.jl")
    include("update.jl")
    include("problems.jl")
end