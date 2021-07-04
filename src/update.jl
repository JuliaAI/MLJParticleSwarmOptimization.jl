# Move the swarm

function move!(state::ParticleSwarmState{T}, ps::ParticleSwarm) where {T}
    rng, X, V = ps.rng, state.X, state.V
    w, c1, c2 = T.((ps.w, ps.c1, ps.c2))
    V .= w.*V .+ c1.*rand.(Ref(rng)).*(state.pbest_X .- X) .+
                 c2.*rand.(Ref(rng)).*(state.gbest_X .- X)
    X .+= V
    for (r, idx) in zip(state.ranges, state.indices)
        constrain!(r, view(X, :, idx))
    end
    return state
end

# Constrain particles' positions

function constrain!(r::NominalRange, X::AbstractArray{T}) where {T}
    @. X = min(one(T), max(eps(T), X))
    return X ./= sum(X, dims=2)
end

function constrain!(r::NumericRange, X::AbstractArray{T}) where {T}
    return @. X = min(T(r.upper), max(T(r.lower), X))
end

# Update pbest

function pbest!(state::ParticleSwarmState, ps::ParticleSwarm, measurements)
    X, pbest, pbest_X = state.X, state.pbest, state.pbest_X
    prob_shift = ps.prob_shift
    improved = measurements .<= pbest
    pbest[improved] .= measurements[improved]
    for (r, p, i) in zip(state.ranges, state.parameters, state.indices)
        @views _pbest!(pbest_X[improved, i], X[improved, i], r, p[improved], prob_shift)
    end
    return state
end

_pbest!(pbest_X, X, r::ParamRange, p, prob_shift) = pbest_X .= X

function _pbest!(pbest_X::AbstractArray{T}, X, r::NominalRange, p, prob_shift) where {T}
    pbest_X .= X .* (one(T) - T(prob_shift))
    sampled = map(pᵢ -> findfirst(==(pᵢ), r.values), p)
    for (i, j) in zip(axes(pbest_X, 1), sampled)
        pbest_X[i, j] += T(prob_shift)
    end
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