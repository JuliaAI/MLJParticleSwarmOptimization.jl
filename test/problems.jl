@testset "Optimize" begin
    function optimize(f, ranges, ps::ParticleSwarm; iter=100)
        fields = getproperty.(ranges, :field)
        state = PSO.initialize(ranges, ps)
        for i in 1:iter
            PSO.move!(state, ps)
            PSO.retrieve!(state, ps)
            measurements = [f(zip(fields, params)) for params in zip(state.parameters...)]
            PSO.pbest!(state, ps, measurements)
            PSO.gbest!(state, ps)
        end
        min, particle = findmin(state.pbest)
        return min, state.pbest_X[particle, :]
    end

    @testset "Ackley" begin
        function ackley(x; a=20, b=0.2, c=2π)
            d = length(x)
            return -a * exp(-b * sqrt(sum(x.^2) / d)) -
                exp(sum(cos.(c .* x)) / d) + a + ℯ
        end

        @testset "Numeric Ackley" begin
            r1 = range(Float64, :r1; lower=-2.0, upper=2.0)
            r2 = range(Float64, :r2; lower=-2.0, upper=2.0)
            ps = ParticleSwarm(10; rng=StableRNG(1234))
            min, params = optimize([r1, r2], ps) do pairs
                x = [param for (field, param) in pairs]
                ackley(x)
            end
            @test min == 0.025024426526485843
            @test params == [0.007929194115617544; -0.0021395968528412634]
        end

        @testset "Integer Ackley" begin
            r1 = range(Int, :r1; lower=-5, upper=5)
            r2 = range(Int, :r2; lower=-5, upper=5)
            ps = ParticleSwarm(10; rng=StableRNG(1234))
            min, params = optimize([r1, r2], ps) do pairs
                x = [param for (field, param) in pairs]
                ackley(x)
            end
            @test min == 4.440892098500626e-16
            @test params == [0.02975036379282736; 0.49855634876840194]
        end
    end
end
