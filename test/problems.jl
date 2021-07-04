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
            ps = ParticleSwarm(n_particles=10, rng=StableRNG(1234))
            min, params = optimize([r1, r2], ps) do pairs
                x = [param for (field, param) in pairs]
                ackley(x)
            end
            @test min == 0.025024426526485843
            @test params == [0.007929194115617544; -0.0021395968528412634]
        end

        @testset "Integer Ackley" begin
            r1 = range(Int, :r1; lower=-20, upper=20)
            r2 = range(Int, :r2; lower=-20, upper=20)
            ps = ParticleSwarm(n_particles=10, rng=StableRNG(1234))
            min, params = optimize([r1, r2], ps) do pairs
                x = [param for (field, param) in pairs]
                ackley(x)
            end
            @test min == 4.440892098500626e-16
            @test params == [-0.08146121382756652, -0.1797998643479164]
        end

        @testset "Nominal Ackley" begin
            vals = shuffle(StableRNG(1234), -2:2)
            r1 = range(Int, :r1; values=vals)
            r2 = range(Int, :r2; values=vals)
            ps = ParticleSwarm(n_particles=10, rng=StableRNG(1234))
            min, params = optimize([r1, r2], ps) do pairs
                x = [param for (field, param) in pairs]
                ackley(x)
            end
            @test min == 4.440892098500626e-16
            @test params == [
                0.9999999999999993
                1.6653345369377333e-16
                1.6653345369377333e-16
                1.6653345369377333e-16
                1.6653345369377333e-16
                0.9999999999999993
                1.6653345369377333e-16
                1.6653345369377333e-16
                1.6653345369377333e-16
                1.6653345369377333e-16
            ]
        end

        @testset "Mixed Ackley" begin
            vals = shuffle(StableRNG(1234), -2:2)
            r1 = range(Float64, :r1; lower=-2.0, upper=2.0)
            r2 = range(Int, :r2; lower=-20, upper=20)
            r3 = range(Int, :r3; values=vals)
            ps = ParticleSwarm(n_particles=10, rng=StableRNG(1234))
            min, params = optimize([r1, r2, r3], ps; iter=1000) do pairs
                x = [param for (field, param) in pairs]
                ackley(x)
            end
            @test min == 0.020697608715345428
            @test params == [
                 0.008417879502700232
                -0.24827995845277284
                 0.6479650806289863
                 8.836599910222148e-17
                 8.836599910222148e-17
                 0.35203491937101344
                 8.836599910222148e-17
            ]
        end
    end
end
