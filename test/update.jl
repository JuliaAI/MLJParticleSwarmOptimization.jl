@testset "Update" begin
    r1 = range(String, :r1; values=["a", "b", "c"])
    r2 = range(Int, :r2; lower=1, upper=3, scale=exp10)
    r3 = range(Float64, :r3; lower=1, upper=Inf, origin=4, unit=1)
    r4 = range(Float32, :r4; lower=-Inf, upper=Inf, origin=0, unit=1)

    @testset "Move Swarm" begin
        X = [0.14280010160187237 0.49409071076694583 0.3631091876311819  2.7429797605672808 3.9372495283243105 -0.8067647083847199;
             0.295010672512568   0.4534584876713112  0.25153083981612073 2.3976392099947    3.6569395920512977  0.420991611378423 ;
             0.12881930462550284 0.2617407494915029  0.6094399458829942  2.5742724788985445 3.6354556967115146  0.673601904658013 ]
        gbest_X = [-0.8571998983981276   0.49409071076694583  1.3631091876311818  1.7429797605672808  7.93724952832431   -0.8067647083847199;
                    1.295010672512568   -0.5465415123286887   0.25153083981612073 5.397639209994701   3.6569395920512977 -99.57900838862157 ;
                    0.12881930462550284  1.261740749491503   -0.3905600541170058  2.5742724788985445 -0.3645443032884854  100.67360190465801]
        dummy_state = PSO.ParticleSwarmState(
            (r1, r2, r3, r4),
            nothing,
            (1:3, 4, 5, 6),
            X,
            zeros(3, 6),
            X,
            gbest_X,
            Float64[],
            Float64[]
        )
        ps = ParticleSwarm(n_particles=3, rng=StableRNG(8888))
        PSO.move!(dummy_state, ps)

        @test all(0 .<= X[:, 1:3] .<= 1)
        @test all(sum(X[:, 1:3]; dims=2) .== 1)
        @test all(r2.lower .<= X[:, 4] .<= r2.upper)
        @test all(r3.lower .<= X[:, 5] .<= r3.upper)
        @test all(r4.lower .<= X[:, 6] .<= r4.upper)
        @test X ≈ [1.4861521012405697e-16 0.3306965950637089  0.6693034049362909    2.426443736254841  9.889381738850881  -0.8067647083847199;
                   0.6561834218566733     0.17876621097030176 0.16505036717302488   3.0                3.6569395920512977 -40.736143561117814;
                   0.11411862296972314    0.8858813770302767  1.967051803731305e-16 2.5742724788985445 1.0                 73.175744055348   ]
    end

    ps = ParticleSwarm(n_particles=3, rng=StableRNG(1234))
    state = PSO.initialize([r1, r2, r3, r4], ps)
    PSO.retrieve!(state, ps)
    measurements = [0.5, 0.1, 0.3]
    PSO.pbest!(state, ps, measurements)
    PSO.gbest!(state, ps)

    @testset "Update personal best" begin
        @test state.pbest == measurements
        @test state.pbest_X ≈ [0.35710007620140427 0.37056803307520936 0.2723318907233864  2.7429797605672808 3.9372495283243105 -0.8067647083847199;
                               0.471258004384426   0.3400938657534834  0.18864812986209056 2.3976392099947    3.6569395920512977  0.420991611378423 ;
                               0.09661447846912713 0.19630556211862715 0.7070799594122457  2.5742724788985445 3.6354556967115146  0.6736019046580138]
    end

    @testset "Update global best" begin
        @test all(state.gbest .== 0.1)
        @test all(state.gbest_X .== state.pbest_X[2, :]')
    end
end