struct ParticleSwarm{R<:AbstractRNG}
    n_particles::Int
    rng::R
    # TODO: topology

    function ParticleSwarm(n_particles, rng::R=Random.GLOBAL_RNG) where {R}
        n_particles < 3 && throw(ArgumentError("There must be at least 3 particles."))
        return new{R}(n_particles, rng)
    end
end

struct ParticleSwarmState
    R::Vector{ParamRange}
    I::Vector{UnitRange{Int}}
    X::Matrix{Float64}
    V::Matrix{Float64}
    pbest_X::Matrix{Float64}
    gbest_X::Matrix{Float64}
    pbest::Vector{Float64}
    gbest::Vector{Float64}
end