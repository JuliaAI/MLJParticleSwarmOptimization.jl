# Move the swarm

function move!(state::ParticleSwarmState{T}, ps::ParticleSwarm) where {T}
    rng, X, V = ps.rng, state.X, state.V
    w, c1, c2 = T.(coefficients(ps.coeffs))
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
    @. X = min(one(T), max(eps(T), X))
    return X ./= sum(X, dims=2)
end

constrain!(r::NumericRange, X) = @. X = min(r.upper, max(r.lower, X))

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

function _pbest!(pbest_X, X, r::NominalRange, p, prob_shift)
    pbest_X .= X .* prob_shift
    samples = map(pᵢ -> findfirst(pᵢ .== r.values), p)
    pbest_X[CartesianIndex.(axes(pbest_X, 1), samples)] .+= 1 - prob_shift
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