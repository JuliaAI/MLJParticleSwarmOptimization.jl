struct ParticleSwarm{C, R<:AbstractRNG}
    n_particles::Int
    coefficients::C
    rng::R
    # TODO: topology

    function ParticleSwarm{C, R}(n_particles, coefficients, rng) where {C, R}
        n_particles < 3 && throw(ArgumentError("There must be at least 3 particles."))
        return new(n_particles, coefficients, rng)
    end
end

function ParticleSwarm(
    n_particles=3;
    coefficients::C=(1., 2., 2.),
    rng::R=Random.GLOBAL_RNG
) where {C, R}
    return ParticleSwarm{C, R}(n_particles, coefficients, rng)
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