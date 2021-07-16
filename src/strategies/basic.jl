"""
    ParticleSwarm(n_particles = 3,
                  w = 1.0,
                  c1 = 2.0,
                  c2 = 2.0,
                  prob_shift = 0.25,
                  rng = Random.GLOBAL_RNG)

Instantiate a particle swarm optimization tuning strategy. A swarm is initiated
by sampling hyperparameters with their customizable priors, and new models are
generated by referencing each member's and the swarm's best models so far.

### Supported ranges

A single one-dimensional range or vector of one-dimensional ranges can be
specified. `ParamRange` objects are constructed using the `range` method. If not
paired with a prior, then one is fitted, as follows:

| Range Types             | Default Distribution |
|:----------------------- |:-------------------- |
| `NominalRange`          | `Dirichlet`          |
| Bounded `NumericRange`  | `Uniform`            |
| Positive `NumericRange` | `Gamma`              |
| Other `NumericRange`    | `Normal`             |

Specifically, in `ParticleSwarm`, the `range` field of a `TunedModel` instance
can be:

- a single one-dimensional range (`ParamRange` object) `r`

- a pair of the form `(r, d)`, with `r` as above and where `d` is:

    - a Dirichlet distribution with the same number of categories as `r.values`
      (for `NominalRange` `r`)

    - any `Distributions.UnivariateDistribution` *instance* (for `NumericRange`
      `r`)

    - one of the distribution *types* in the table below, for automatic fitting
      using `Distributions.fit(d, r)` to a distribution whose support always
      lies between `r.lower` and `r.upper` (for `NumericRange` `r`) or the set
      of probability vectors (for `NominalRange` `r`)

- any vector of objects of the above form

| Range Types             | Distribution Types                                                                           |
|:----------------------- |:-------------------------------------------------------------------------------------------- |
| `NominalRange`          | `Dirichlet`                                                                                  |
| Bounded `NumericRange`  | `Arcsine`, `Uniform`, `Biweight`, `Cosine`, `Epanechnikov`, `SymTriangularDist`, `Triweight` |
| Positive `NumericRange` | `Gamma`, `InverseGaussian`, `Poisson`                                                        |
| Any `NumericRange`      | `Normal`, `Logistic`, `LogNormal`, `Cauchy`, `Gumbel`, `Laplace`                             |

### Examples

    using Distributions

    range1 = range(model, :hyper1, lower=0, upper=1)

    range2 = [(range(model, :hyper1, lower=1, upper=10), Arcsine),
              range(model, :hyper2, lower=2, upper=Inf, unit=1, origin=3),
              (range(model, :hyper2, lower=2, upper=4), Normal(0, 3)),
              (range(model, :hyper3, values=[:ball, :tree]), Dirichlet)]

### Algorithm

Hyperparameter ranges are sampled and concatenated into position vectors for
each swarm particle. Velocity is initiated to be zeros, and in each iteration,
every particle's position is updated to approach its personal best and the
swarm's best models so far with the equations:

\$vₖ₊₁ = w⋅vₖ + c₁⋅rand()⋅(pbest - x) + c₂⋅rand()⋅(gbest - x)\$

\$xₖ₊₁ = xₖ + vₖ₊₁\$

New models are then generated for evaluation by mutating the fields of a deep
copy of `model`. If the corresponding range has a specified `scale` function,
then the transformation is applied before the hyperparameter is returned. For
integer `NumericRange`s, the hyperparameter is rounded; and for `NominalRange`s,
the hyperparameter is sampled from the specified values with the probability
weights given by each particle.

Personal and social best models are then updated for the swarm. In order to
replicate both the probability weights and the sampled value for `NominalRange`s
of the best models, the weights of unselected values are shifted to the selected
one by the `prob_shift` factor.
"""
mutable struct ParticleSwarm{R<:AbstractRNG} <: AbstractParticleSwarm
    n_particles::Int
    w::Float64
    c1::Float64
    c2::Float64
    prob_shift::Float64
    rng::R
    # TODO: topology
end

function ParticleSwarm(;
    n_particles=3,
    w=1.0,
    c1=2.0,
    c2=2.0,
    prob_shift=0.25,
    rng::R=Random.GLOBAL_RNG
) where {R}
    swarm = ParticleSwarm{R}(n_particles, w, c1, c2, prob_shift, rng)
    message = MLJTuning.clean!(swarm)
    isempty(message) || @warn message
    return swarm
end

function MLJTuning.clean!(tuning::ParticleSwarm)
    warning = ""
    if tuning.n_particles < 3
        warning *= "ParticleSwarm requires at least 3 particles. Resetting n_particles=3. "
        tuning.n_particles = 3
    end
    if tuning.w < 0
        warning *= "ParticleSwarm requires w ≥ 0. Resetting w=1. "
        tuning.w = 1
    end
    if tuning.c1 < 0
        warning *= "ParticleSwarm requires c1 ≥ 0. Resetting c1=2. "
        tuning.c1 = 2
    end
    if tuning.c2 < 0
        warning *= "ParticleSwarm requires c2 ≥ 0. Resetting c2=2. "
        tuning.c2 = 2
    end
    if !(0 ≤ tuning.prob_shift < 1)
        warning *= "ParticleSwarm requires 0 ≤ prob_shift < 1. Resetting prob_shift=0.25. "
        tuning.prob_shift = 0.25
    end
    return warning
end

function MLJTuning.setup(tuning::ParticleSwarm, model, ranges, n, verbosity)
    return initialize(ranges, tuning)
end

function MLJTuning.models(
    tuning::ParticleSwarm,
    model,
    history,
    state,
    n_remaining,
    verbosity
)
    n_particles = tuning.n_particles
    if !isnothing(history)
        sig = MLJTuning.signature(first(history).measure)
        pbest!(state, tuning, map(h -> sig * h.measurement[1],
                                  history[end-n_particles+1:end]))
        gbest!(state, tuning)
        move!(state, tuning)
    end
    retrieve!(state, tuning)
    fields = getproperty.(state.ranges, :field)
    new_models = map(1:n_particles) do i
        clone = deepcopy(model)
        for (field, param) in zip(fields, getindex.(state.parameters, i))
            recursive_setproperty!(clone, field, param)
        end
        clone
    end
    return new_models, state
end

function MLJTuning.tuning_report(tuning::ParticleSwarm, history, state)
    fields = getproperty.(state.ranges, :field)
    scales = MLJBase.scale.(state.ranges)
    return (; plotting = MLJTuning.plotting_report(fields, scales, history))
end