mutable struct ParticleSwarm{C, P, R<:AbstractRNG} <: MLJTuning.TuningStrategy
    n_particles::Int
    coeffs::C
    prob_shift::P
    rng::R
    # TODO: topology

    function ParticleSwarm{C, P, R}(n_particles, coeffs, prob_shift, rng) where {C, P, R}
        n_particles < 3 && throw(ArgumentError("There must be at least 3 particles."))
        0 <= prob_shift < 1 || throw(ArgumentError("Probability shift must be in [0, 1)."))
        return new(n_particles, coeffs, prob_shift, rng)
    end
end

function ParticleSwarm(
    n_particles=3;
    coeffs::C=StaticCoeffs(),
    prob_shift::P=0.5,
    rng::R=Random.GLOBAL_RNG
) where {C, P, R}
    return ParticleSwarm{C, P, R}(n_particles, coeffs, prob_shift, rng)
end

function Base.setproperty!(ps::ParticleSwarm, sym::Symbol, val)
    if sym === :n_particles && val < 3
        throw(ArgumentError("There must be at least 3 particles."))
    end
    return setfield!(ps, sym, val)
end

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