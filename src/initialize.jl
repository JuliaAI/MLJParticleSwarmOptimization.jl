# Initialize swarm's positions, velocities, and get lower and upper bounds

function initialize(ps::ParticleSwarm, r::Union{ParamRange, Tuple{ParamRange, Any}})
    return intialize(ps, [r])
end

function initialize(ps::ParticleSwarm, rs::AbstractVector)
    R, lens, X = mapreduce(r -> _initialize(ps, r), (r1, r2) -> vcat.(r1, r2), rs)
    I = _indices(lens)
    V = zero(X)
    pbest_X = similar(X)
    gbest_X = similar(X)
    pbest = Vector{Float64}(undef, ps.n_particles)
    gbest = Vector{Float64}(undef, ps.n_particles)
    return ParticleSwarmState(R, I, X, V, pbest_X, gbest_X, pbest, gbest)
end

_initialize(ps, t::Tuple{Union{ParamRange, Any}}) = _initialize(ps, t[1], t[2])

# Initialize hyperparameters with default distribution types

_initialize(ps, r::ParamRange) = _initialize(ps, r, _initializer(r))

_initializer(::NominalRange) = Dirichlet

_initializer(r::NumericRange) = _initializer(MLJTuning.boundedness(r))

_initializer(::Type{MLJBase.Bounded}) = Uniform

_initializer(::Type{MLJTuning.PositiveUnbounded}) = Gamma

_initializer(::Type{MLJTuning.Other}) = Normal

# Fit distribution and initialize hyperparameter

function _initialize(ps, r::ParamRange, D::Type{<:Distribution})
    throw(ArgumentError("$D distribution is unsupported for $(typeof(r))."))
end

function _initialize(ps, r::NominalRange, D::Type{Dirichlet})
    d = Dirichlet(ones(length(r.values)))
    return _initialize(ps, r, d)
end

function _initialize(ps, r::NumericRange, D::Type{<:UnivariateDistribution})
    d = Distributions.fit(D, r)
    return _initialize(ps, r, d)
end

# Initialize hyperparameter with fitted/provided distribution

function _initialize(ps, r::ParamRange, d::Distribution)
    throw(ArgumentError("$(typeof(d)) distribution is unsupported for $(typeof(r))."))
end

function _initialize(ps, r::NominalRange, d::Dirichlet)
    n = length(r.values)
    n != d.alpha0 &&
        throw(ArgumentError("Provided distribution's number of categories don't match $r."))
    x = convert(Matrix{Float64}, rand(ps.rng, d, ps.n_particles))
    return r, n, x
end

function _initialize(ps, r::NumericRange, d::UnivariateDistribution)
    x = convert(Matrix{Float64}, rand(ps.rng, d, 1, ps.n_particles))
    l = Float64[r.lower]
    u = Float64[r.upper]
    return r, 1, x
end

# Get ranges' corresponding indices

function _indices(lens)
    indices = CartesianIndices{1, Tuple{UnitRange{Int}}}[]
    start = 1
    for len in lens
        stop = start + len - 1
        push!(indices, CartesianIndices((start:stop,)))
        start = stop + 1
    end
    return indices
end