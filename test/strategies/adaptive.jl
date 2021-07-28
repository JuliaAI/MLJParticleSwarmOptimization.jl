@testset "AdaptiveParticleSwarm Tuning Parameters" begin
    warning = "AdaptiveParticleSwarm requires at least 3 particles. " *
              "Resetting n_particles=3. " *
              "AdaptiveParticleSwarm requires 1.5 ≤ c1 ≤ 2.5, 1.5 ≤ c2 ≤ 2.5, and " *
              "c1 + c2 ≤ 4. Resetting coefficients c1=2.0, c2=2.0. " *
              "AdaptiveParticleSwarm requires 0 ≤ prob_shift < 1. " *
              "Resetting prob_shift=0.25. "
    ps = @test_logs (:warn, warning) AdaptiveParticleSwarm(
        n_particles=2, w=1, c1=3, c2=3, prob_shift=2
    )
    @test ps.n_particles == 3
    @test ps.w == 1.0
    @test ps.c1 == 2.0
    @test ps.c2 == 2.0
    @test ps.prob_shift == 0.25
end

for acceleration in (CPU1(), CPUProcesses(), CPUThreads())
    @testset "EvoTree Tuning with AdaptiveParticleSwarm and $(typeof(acceleration))" begin
        rng = StableRNG(123)
        features = rand(rng, 10_000) .* 5 .- 2
        X = MLJBase.table(reshape(features, (size(features)[1], 1)))
        y = sin.(features) .* 0.5 .+ 0.5
        y = EvoTrees.logit(y) + randn(rng, size(y))
        y = EvoTrees.sigmoid(y)

        tree = EvoTreeRegressor(rng=rng)
        r1 = range(tree, :max_depth; values=[3:7;])
        r2 = range(tree, :η; lower=-2, upper=0, scale=exp10)

        baseline_self_tuning_tree = TunedModel(
            model=tree,
            tuning=RandomSearch(rng=StableRNG(1234)),
            # tuning=ParticleSwarm(n_particles=3, rng=rng),
            resampling=CV(nfolds=5, rng=StableRNG(8888)),
            range=[r1, r2],
            measure=(ŷ, y) -> mean(abs.(ŷ .- y)),
            n=15,
            acceleration=acceleration
        )
        baseline_mach = machine(baseline_self_tuning_tree, X, y)
        fit!(baseline_mach, verbosity=0)
        baseline_rep = report(baseline_mach)
        baseline_best_loss = baseline_rep.best_history_entry.measurement[1]

        self_tuning_tree = TunedModel(
            model=tree,
            tuning=AdaptiveParticleSwarm(rng=StableRNG(1234)),
            resampling=CV(nfolds=5, rng=StableRNG(8888)),
            range=[r1, r2],
            measure=(ŷ, y) -> mean(abs.(ŷ .- y)),
            n=15,
            acceleration=acceleration
        )
        mach = machine(self_tuning_tree, X, y)
        fit!(mach, verbosity=0)
        rep = report(mach)
        best_loss = rep.best_history_entry.measurement[1]

        # Compare with random search result with the same settings
        @test best_loss < baseline_best_loss ||
              isapprox(best_loss, baseline_best_loss, 1e-3)
    end
end