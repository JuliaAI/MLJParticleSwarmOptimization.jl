function MLJTuning.setup(tuning::ParticleSwarm, model, range, n, verbosity)
    return initialize(range, tuning)
end

function MLJTuning.models(
    tuning::ParticleSwarm,
    model,
    history,
    state,
    n_remaining,
    verbosity
)
    pbest!(state, tuning, last(history).measurement[1])
    gbest!(state, tuning)
    move!(state, tuning)
    retrieve!(state, tuning)
    fields = getindex.(state.ranges, :field)
    new_models = map(1:ps.n_particles) do i
        clone = deepcopy(model)
        for (field, param) in zip(fields, getindex.(state.parameters, i))
            recursive_setproperty!(clone, field, param)
        end
        clone
    end
    return new_models, state
end