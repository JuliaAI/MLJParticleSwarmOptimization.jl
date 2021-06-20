# Update pbest

function pbest!(state::ParticleSwarmState, ps::ParticleSwarm, measurements)
    X, pbest, pbest_X = state.X, state.pbest, state.pbest_X
    improved = measurements .<= pbest
    pbest[improved] .= measurements
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