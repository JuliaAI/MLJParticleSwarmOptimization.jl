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
        pbest!(state, tuning, map(h -> sig * h.measurement[1], last(history, n_particles)))
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