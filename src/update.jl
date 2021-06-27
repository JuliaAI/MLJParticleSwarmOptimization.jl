# Move the swarm

function move!(state::ParticleSwarmState{T}, ps::ParticleSwarm) where {T}
    rng, X, V = ps.rng, state.X, state.V
    w, c1, c2 = T.(coefficients(ps.coefficients))
    V .= w.*V .+ c1.*rand.(Ref(rng)).*(state.pbest_X .- X) .+
                 c2.*rand.(Ref(rng)).*(state.gbest_X .- X)
    X .+= V
    for (r, idx) in zip(state.ranges, state.indices)
        constrain!(r, view(X, :, idx))
    end
    return state
end

# Constrain particles' positions

function constrain!(r::NominalRange, X)
    T = eltype(X)
    @. X = min(one(T), max(zero(T), X))
    return X ./= sum(X, dims=2)
end

constrain!(r::NumericRange, X) = @. X = min(r.upper, max(r.lower, X))

# Update pbest

function pbest!(state::ParticleSwarmState, ps::ParticleSwarm, measurements)
    X, pbest, pbest_X = state.X, state.pbest, state.pbest_X
    improved = measurements .<= pbest
    pbest[improved] .= measurements[improved]
    for (r, p, i) in zip(state.ranges, state.parameters, state.indices)
        _shift!(view(pbest_X, improved, i), view(X, improved, i), r, p)
    end
    return state
end

_shift!(pbest_X, X, r::ParamRange, p) = pbest_X .= X

function _shift!(pbest_X, X, r::NominalRange, p)
    T = eltype(X)
    pbest_X .= X .* T(0.75)
    samples = map(pᵢ -> findfirst(pᵢ .== r.values), p)
    pbest_X[CartesianIndex.(axes(pbest_X, 1), samples)] .+= T(0.25)
    return pbest_X
end

# Update gbest

function gbest!(state::ParticleSwarmState, ps::ParticleSwarm)
    pbest, pbest_X, gbest, gbest_X = state.pbest, state.pbest_X, state.gbest, state.gbest_X
    best, i = findmin(pbest)
    gbest .= best
    gbest_X .= pbest_X[i, :]'
    return state
end