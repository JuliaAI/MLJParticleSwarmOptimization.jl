# Retrieve parameters

function retrieve(ps::ParticleSwarm, state::ParticleSwarmState)
    R, I, X = state.R, state.I, state.X
    return zip((_retrieve(r, view(X, idx, :)) for (r, idx) in zip(R, I))...)
end

function _retrieve(r::NominalRange, X)
    return (sample([r.values...], Weights(probs)) for probs in eachcol(X))
end

_retrieve(r::NumericRange{T}, X) where {T<:Real} = (T(_transform(r.scale, x)) for x in X[:])

function _retrieve(r::NumericRange{T}, X) where {T<:Integer}
    return (round(T, _transform(r.scale, x)) for x in X[:])
end

_transform(::Symbol, x) = x

_transform(scale, x) = scale(x)

# Update pbest and gbest

function pbest!(ps::ParticleSwarm, state::ParticleSwarmState, measurements)
    X, pbest, pbest_X = state.X, state.pbest, state.pbest_X
    improved = measurements .<= pbest
    pbest[improved] .= measurements
    pbest_X[:, improved] .= X[:, improved]
    return state
end

function gbest!(ps::ParticleSwarm, state::ParticleSwarmState)
    pbest, pbest_X, gbest, gbest_X = state.pbest, state.pbest_X, state.gbest, state.gbest_X
    best, idx = findmin(pbest)
    gbest .= best
    gbest_X .= pbest_X[:, idx]
    return state
end

# Move swarm

function move!(ps::ParticleSwarm, state::ParticleSwarmState)
    rng, X, V = ps.rng, state.X, state.V
    w, c1, c2 = coefficients(ps)
    @. V = w*V + c1*rand(rng)*(state.pbest_X - X) + c2*rand(rng)*(state.gbest_X - X)
    X .+= V
    for (r, idx) in zip(state.R, state.I)
        _constrain!(r, view(X, idx, :))
    end
    return state
end

function _constrain!(r::NominalRange, x)
    @. x = min(1.0, max(0.0, x))
    return x ./ sum(x, dims=1)
end

_constrain!(r::NumericRange, x) = @. x = min(r.upper, max(r.lower, x))