abstract type AbstractParticleSwarm <: MLJTuning.TuningStrategy end

struct ParticleSwarmState{T, R, P, I}
    ranges::R
    parameters::P
    indices::I
    X::Matrix{T}
    V::Matrix{T}
    pbest_X::Matrix{T}
    gbest_X::Matrix{T}
    pbest::Vector{T}
    gbest::Vector{T}
end

mutable struct ParticleSwarm <: AbstractParticleSwarm
    n_particles::Integer
    w::Float64
    c1::Float64
    c2::Float64
    prob_shift::Float64
    rng::AbstractRNG
    # TODO: topology
end

mutable struct AdaptiveParticleSwarm <: AbstractParticleSwarm
    n_particles::Integer
    c1::Float64
    c2::Float64
    prob_shift::Float64
    rng::AbstractRNG
end

get_n_particles(tuning::AbstractParticleSwarm) = tuning.n_particles
get_prob_shift(tuning::AbstractParticleSwarm) = tuning.prob_shift
get_rng(tuning::AbstractParticleSwarm) = tuning.rng

function initialize(r, tuning::AbstractParticleSwarm)
    return initialize(get_rng(tuning), r, get_n_particles(tuning))
end

function retrieve!(state::ParticleSwarmState, tuning::AbstractParticleSwarm)
    return retrieve!(get_rng(tuning), state)
end

function pbest!(state::ParticleSwarmState, measurements, tuning::AbstractParticleSwarm)
    return pbest!(state, measurements, get_prob_shift(tuning))
end