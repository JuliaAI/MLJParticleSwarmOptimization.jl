@testset "Swarm Parameters" begin
    warning = "ParticleSwarm requires at least 3 particles. Resetting n_particles=3. " *
              "ParticleSwarm requires w ≥ 0. Resetting w=1. " *
              "ParticleSwarm requires c1 ≥ 0. Resetting c1=2. " *
              "ParticleSwarm requires c2 ≥ 0. Resetting c2=2. " *
              "ParticleSwarm requires 0 ≤ prob_shift < 1. Resetting prob_shift=0.25. "
    ps = @test_logs (:warn, warning) ParticleSwarm(
        n_particles=2, w=-1, c1=-1, c2=-1, prob_shift=2
    )
    @test ps.n_particles == 3
    @test ps.w == 1.0
    @test ps.c1 == 2.0
    @test ps.c2 == 2.0
    @test ps.prob_shift == 0.25
end

@testset "EvoTree Tuning" begin
    rng = StableRNG(1234)
    features = rand(rng, 10_000) .* 5 .- 2
    X = MLJBase.table(reshape(features, (size(features)[1], 1)))
    y = sin.(features) .* 0.5 .+ 0.5
    y = EvoTrees.logit(y) + randn(rng, size(y))
    y = EvoTrees.sigmoid(y)

    tree = EvoTreeRegressor(rng=rng)
    r1 = range(tree, :max_depth; values=[3:7;])
    r2 = range(tree, :λ; lower=0.0, upper=Inf, origin=0.25, unit=0.1)
    r3 = range(tree, :η; lower=-2, upper=0, scale=exp10)
    self_tuning_tree = TunedModel(
        model=tree,
        tuning=RandomSearch(rng=StableRNG(1234)),
        # tuning=ParticleSwarm(n_particles=3, rng=rng),
        resampling=CV(nfolds=5),
        range=[r1, r2, r3],
        measure=(ŷ, y) -> mean(abs.(ŷ .- y)),
        n=15
    )
    mach = machine(self_tuning_tree, X, y)
    fit!(mach, verbosity=0)

    rep = report(mach)
    best_model = rep.best_model
    # Compare with random search result with the same settings
    @test rep.best_history_entry.measurement[1] ≤ 0.08951616508383785
end