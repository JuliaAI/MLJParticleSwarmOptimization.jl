# Move the swarm

function move!(state::ParticleSwarmState{T}, ps::ParticleSwarm) where {T}
    rng, X, V = ps.rng, state.X, state.V
    w, c1, c2 = T.(coefficients(ps.coefficients))
    @. V = w*V + c1*rand(rng)*(state.pbest_X - X) + c2*rand(rng)*(state.gbest_X - X)
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

# Coefficients schemes

coefficients(coeffs::Union{Tuple, NamedTuple}) = coeffs

# TODO: Add adaptive coefficients