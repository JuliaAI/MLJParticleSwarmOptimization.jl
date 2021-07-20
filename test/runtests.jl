using ParticleSwarmOptimization
using Random
using Test
using ComputationalResources
using Distributions
using EvoTrees
using MLJBase
using MLJTuning
using StableRNGs

const PSO = ParticleSwarmOptimization

include("parameters.jl")
include("update.jl")
include("problems.jl")
include("strategies/basic.jl")