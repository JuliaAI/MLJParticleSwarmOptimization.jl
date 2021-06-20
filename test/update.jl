@testset "Update" begin
    r1 = range(String, :r1; values=["a", "b", "c"])
    r2 = range(Int, :r2; lower=1, upper=3, scale=exp10)
    r3 = range(Float64, :r3; lower=1, upper=Inf, origin=4, unit=1)
    r4 = range(Float32, :r4; lower=-Inf, upper=Inf, origin=0, unit=1)
    ps = ParticleSwarm(3; rng=StableRNG(1234))
    state = PSO.initialize([r1, r2, r3, r4], ps)
    PSO.retrieve!(state, ps)
    measurements = [0.5, 0.1, 0.3]
    PSO.pbest!(state, ps, measurements)
    PSO.gbest!(state, ps)

    @testset "Update personal best" begin
        @test state.pbest == measurements
        @test state.pbest_X == [0.35710007620140427 0.37056803307520936 0.2723318907233864  2.7429797605672808 3.9372495283243105 -0.8067647083847199;
                                0.471258004384426   0.3400938657534834  0.18864812986209056 2.3976392099947    3.6569395920512977  0.420991611378423 ;
                                0.09661447846912713 0.19630556211862715 0.7070799594122457  2.5742724788985445 3.6354556967115146  0.6736019046580138]
    end

    @testset "Update global best" begin
        @test all(state.gbest .== 0.1)
        @test all(state.gbest_X .== state.pbest_X[2, :]')
    end
end