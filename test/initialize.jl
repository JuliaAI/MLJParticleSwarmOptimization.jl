@testset "Initialize" begin
    r1 = range(String, :r1; values=["a", "b", "c"])
    r2 = range(Int, :r2; lower=1, upper=3)
    r3 = range(Float64, :r3; lower=1, upper=Inf, origin=4, unit=1)
    r4 = range(Float32, :r4; lower=-Inf, upper=Inf, origin=0, unit=1)
    d1 = Dirichlet(ones(3))
    d2 = Uniform(1, 3)
    d3 = truncated(Gamma(16, 0.25), 1, Inf)
    d4 = Normal(0, 1)
    X1 = [0.14280010160187237 0.295010672512568   0.12881930462550284;
          0.49409071076694583 0.4534584876713112  0.2617407494915029;
          0.3631091876311819  0.25153083981612073 0.6094399458829942]
    X2 = [2.7429797605672808  2.3976392099947     2.5742724788985445]
    X3 = [3.9372495283243105  3.6569395920512977  3.6354556967115146]
    X4 = [-0.8067647083847199 0.420991611378423   0.6736019046580138]
    I = [1:3, 4:4, 5:5, 6:6]

    @testset "Initializer" begin
        @test PSO._initializer(r1) === Dirichlet
        @test PSO._initializer(r2) === Uniform
        @test PSO._initializer(r3) === Gamma
        @test PSO._initializer(r4) === Normal 
    end

    @testset "Initialize with distribution types" begin
        ps = ParticleSwarm(3, StableRNG(1234))
        @test PSO._initialize(ps, r1, Dirichlet) == (r1, 3, X1)
        @test PSO._initialize(ps, r2, Uniform) == (r2, 1, X2)
        @test PSO._initialize(ps, r3, Gamma) == (r3, 1, X3)
        @test PSO._initialize(ps, r4, Normal) == (r4, 1, X4)
    end

    @testset "Initialize with distributions" begin
        ps = ParticleSwarm(3, StableRNG(1234))
        @test PSO._initialize(ps, r1, d1) == (r1, 3, X1)
        @test PSO._initialize(ps, r2, d2) == (r2, 1, X2)
        @test PSO._initialize(ps, r3, d3) == (r3, 1, X3)
        @test PSO._initialize(ps, r4, d4) == (r4, 1, X4)
    end

    @testset "Range Indices" begin
        @test PSO._indices([3,1,1,1]) == I
    end

    @testset "Unsupported distributions" begin
        ps = ParticleSwarm(3, StableRNG(1234))
        @test_throws ArgumentError PSO._initialize(ps, r1, Uniform)
        @test_throws ArgumentError PSO._initialize(ps, r1, d2)
        @test_throws ArgumentError PSO._initialize(ps, r1, Dirichlet(ones(4)))
        @test_throws ArgumentError PSO._initialize(ps, r2, Dirichlet)
        @test_throws ArgumentError PSO._initialize(ps, r2, d1)
    end

    @testset "Initialize one range" begin
        ps = ParticleSwarm(3, StableRNG(1234))
        state = PSO.initialize(ps, r1)
        @test state.R == MLJBase.ParamRange[r1]
        @test state.I == [1:3]
        @test state.X == X1
    end

    @testset "Initialize multiple ranges" begin
        ps = ParticleSwarm(3, StableRNG(1234))
        ranges = [r1, (r2, Uniform), (r3, d3), r4]
        state = PSO.initialize(ps, ranges)
        @test state.R == MLJBase.ParamRange[r1, r2, r3, r4]
        @test state.I == I
        @test state.X == vcat(X1, X2, X3, X4)
    end
end