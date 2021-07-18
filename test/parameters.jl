@testset "Parameters" begin
    # Swarm size
    n = 3
    # Test ranges
    r1 = range(String, :r1; values=["a", "b", "c"])
    r2 = range(Int, :r2; lower=1, upper=3, scale=exp10)
    r3 = range(Float64, :r3; lower=1, upper=Inf, origin=4, unit=1)
    r4 = range(Float32, :r4; lower=-Inf, upper=Inf, origin=0, unit=1)
    rs = (r1, r2, r3, r4)
    # Test distribution types
    Ds = (Dirichlet, Uniform, Gamma, Normal)
    # Manually fitted distributions for test ranges
    d1 = Dirichlet(ones(3))
    d2 = Uniform(1, 3)
    d3 = truncated(Gamma(16, 0.25), 1, Inf)
    d4 = Normal(0, 1)
    ds = (d1, d2, d3, d4)
    # Test range representation lengths
    lengths = (3, 1, 1, 1)
    # Test range's corresponding indices in internal representation state.X
    indices = (1:3, 4, 5, 6)
    # Generated samples from d1, d2, d3, d4 with StableRNG(1234)
    X1 = [0.14280010160187237 0.49409071076694583 0.3631091876311819;
          0.295010672512568   0.4534584876713112  0.25153083981612073;
          0.12881930462550284 0.2617407494915029  0.6094399458829942]
    X2 = [2.7429797605672808, 2.3976392099947, 2.5742724788985445]
    X3 = [3.9372495283243105, 3.6569395920512977, 3.6354556967115146]
    X4 = [-0.8067647083847199, 0.420991611378423, 0.6736019046580138]
    Xs = (X1, X2, X3, X4)

    @testset "Initializer" begin
        for (r, D) in zip(rs, Ds)
            @test PSO._initializer(r) === D
        end
    end

    @testset "Initialize with distribution types" begin
        rng = StableRNG(1234)
        for (r, D, l, X) in zip(rs, Ds, lengths, Xs)
            r̂, l̂, X̂ = PSO._initialize(rng, r, D, n)[[1,3,4]]
            @test r̂ === r
            @test l̂ == l
            @test X̂ ≈ X
        end
    end

    @testset "Initialize with distributions" begin
        rng = StableRNG(1234)
        for (r, d, l, X) in zip(rs, ds, lengths, Xs)
            r̂, l̂, X̂ = PSO._initialize(rng, r, d, n)[[1,3,4]]
            @test r̂ === r
            @test l̂ == l
            @test X̂ ≈ X
        end
    end

    @testset "Range Indices" begin
        @test PSO._to_indices(lengths) == indices
    end

    @testset "Unsupported distributions" begin
        rng = StableRNG(1234)
        @test_throws ArgumentError PSO._initialize(rng, r1, Uniform, n)
        @test_throws ArgumentError PSO._initialize(rng, r1, d2, n)
        @test_throws ArgumentError PSO._initialize(rng, r1, Dirichlet(ones(4)), n)
        @test_throws ArgumentError PSO._initialize(rng, r2, Dirichlet, n)
        @test_throws ArgumentError PSO._initialize(rng, r2, d1, n)
    end

    @testset "Initialize one range" begin
        ps = ParticleSwarm(n_particles=n, rng=StableRNG(1234))
        for (r, l, i, X) in zip(rs, lengths, indices, Xs)
            state = PSO.initialize(r, ps)
            @test state.ranges == (r,)
            @test state.indices == (l == 1 ? 1 : 1:l,)
            @test state.X ≈ X
        end
    end

    @testset "Initialize multiple ranges" begin
        ps = ParticleSwarm(n_particles=n, rng=StableRNG(1234))
        ranges = [r1, (r2, Uniform), (r3, d3), r4]
        state = PSO.initialize(ranges, ps)
        @test state.ranges == rs
        @test state.indices == indices
        @test state.X ≈ hcat(Xs...)
    end

    @testset "Retrieve parameters" begin
        ps = ParticleSwarm(n_particles=n, rng=StableRNG(1234))
        ranges = [r1, (r2, Uniform), (r3, d3), r4]
        state = PSO.initialize(ranges, ps)
        PSO.retrieve!(state, ps)
        @test state.parameters == (
            ["a", "a", "c"],
            [553, 250, 375],
            [3.9372495283243105, 3.6569395920512977, 3.6354556967115146],
            [-0.8067647f0, 0.4209916f0, 0.6736019f0]
        )
    end
end