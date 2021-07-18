###
### Initialization
###

# Initialize particle swarm state

function initialize(
    r::Union{ParamRange, Tuple{ParamRange, Any}},
    tuning::AbstractParticleSwarm
)
    return initialize([r], tuning)
end

function initialize(rs::AbstractVector, tuning::AbstractParticleSwarm)
    n = tuning.n_particles
    ranges, parameters, lens, Xᵢ = zip(_initialize.(tuning.rng, rs, n)...)
    indices = _to_indices(lens)
    X = hcat(Xᵢ...)
    V = zero(X)
    pbest_X = copy(X)
    gbest_X = copy(X)
    pbest = fill(eltype(X)(Inf), n)
    gbest = similar(pbest)
    return ParticleSwarmState(
        ranges, parameters, indices, X, V, pbest_X, gbest_X, pbest, gbest
    )
end

# Unpack tuple of range and distribution

function _initialize(rng, t::Tuple{ParamRange, Any}, n)
    return _initialize(rng, t[1], t[2], n)
end

# Initialize parameters with default distribution types

_initialize(rng, r::ParamRange, n) = _initialize(rng, r, _initializer(r), n)

_initializer(::NominalRange) = Dirichlet

_initializer(r::NumericRange) = _initializer(MLJTuning.boundedness(r))

_initializer(::Type{MLJBase.Bounded}) = Uniform

_initializer(::Type{MLJTuning.PositiveUnbounded}) = Gamma

_initializer(::Type{MLJTuning.Other}) = Normal

# Fit distributions and initialize parameters

function _initialize(rng, r::ParamRange, D::Type{<:Distribution}, n)
    throw(ArgumentError("$D distribution is unsupported for $(typeof(r))."))
end

function _initialize(rng, r::NominalRange{T, N}, D::Type{Dirichlet}, n) where {T, N}
    d = Dirichlet(ones(N))
    return _initialize(rng, r, d, n)
end

function _initialize(rng, r::NumericRange, D::Type{<:UnivariateDistribution}, n)
    d = Distributions.fit(D, r)
    return _initialize(rng, r, d, n)
end

# Initialize parameters with fitted/provided distributions

function _initialize(rng, r::ParamRange, d::Distribution, n)
    throw(ArgumentError("$(typeof(d)) distribution is unsupported for $(typeof(r))."))
end

function _initialize(rng, r::NominalRange{T, N}, d::Dirichlet, n) where {T, N}
    N != d.alpha0 &&
        throw(ArgumentError("Provided distribution's number of categories don't match $r."))
    p = Vector{T}(undef, n)
    X = rand(rng, d, n)'
    return r, p, N, X
end

function _initialize(rng, r::NumericRange{T}, d::UnivariateDistribution, n) where {T}
    p = Vector{T}(undef, n)
    X = rand(rng, d, n)
    return r, p, 1, X
end

# Helper function to get ranges' corresponding indices

function _to_indices(lens)
    curr = Ref(1)
    return map(lens) do len
        start = curr[]
        stop = start + len - 1
        curr[] = stop + 1
        start == stop ? stop : (start:stop)
    end
end

###
### Retrieval
###

function retrieve!(state::ParticleSwarmState, tuning::AbstractParticleSwarm)
    ranges, params, indices, X = state.ranges, state.parameters, state.indices, state.X
    rng = tuning.rng
    for (r, p, i) in zip(ranges, params, indices)
        _retrieve!(rng, p, r, view(X, :, i))
    end
    return state
end

function _retrieve!(rng, p, r::NominalRange, X)
    return p .= getindex.(
        Ref(r.values),
        rand.(rng, Categorical.(X[i,:] for i in axes(X, 1)))
    )
end

function _retrieve!(rng, p, r::NumericRange{T}, X) where {T<:Integer}
    return @. p = round(T, _transform(r.scale, X))
end

function _retrieve!(rng, p, r::NumericRange{T}, X) where {T<:Real}
    return @. p = _transform(r.scale, X)
end

_transform(::Symbol, X) = X

_transform(scale, X) = scale(X)