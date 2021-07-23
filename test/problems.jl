@testset "Optimize" begin
    function optimize(f, ranges, ps::ParticleSwarm, iter::Int)
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
            min, params = optimize([r1, r2], ps, 1_500_000) do pairs
                x = [param for (field, param) in pairs]
                ackley(x)
            end
            @test all(isapprox.(params, 0, atol=3)) # analytical solution
        end

        @testset "Integer Ackley" begin
            r1 = range(Int, :r1; lower=-20, upper=20)
            r2 = range(Int, :r2; lower=-20, upper=20)
            ps = ParticleSwarm(n_particles=10, rng=StableRNG(1234))
            min, params = optimize([r1, r2], ps, 100) do pairs
                x = [param for (field, param) in pairs]
                ackley(x)
            end
            @test all(round.(params) .== 0) # analytical solution
        end

        @testset "Nominal Ackley" begin
            vals = shuffle(StableRNG(1234), -2:2)
            r1 = range(Int, :r1; values=vals)
            r2 = range(Int, :r2; values=vals)
            ps = ParticleSwarm(n_particles=10, rng=StableRNG(1234))
            min, params = optimize([r1, r2], ps, 100) do pairs
                x = [param for (field, param) in pairs]
                ackley(x)
            end
            true_params = [1., 0., 0., 0., 0., 1., 0., 0., 0., 0.] # analytical solution
            @test all(isapprox.(params, true_params, atol=1e-3))
        end

        @testset "Mixed Ackley" begin
            vals = shuffle(StableRNG(1234), -2:2)
            r1 = range(Float64, :r1; lower=-2.0, upper=2.0)
            r2 = range(Int, :r2; lower=-20, upper=20)
            r3 = range(Int, :r3; values=vals)
            ps = ParticleSwarm(n_particles=10, rng=StableRNG(1234))
            min, params = optimize([r1, r2, r3], ps, 15000) do pairs
                x = [param for (field, param) in pairs]
                ackley(x)
            end
            # Compare with analytical solution
            @test isapprox(params[1], 0, atol=1e-3)
            @test round(params[2]) == 0
            @test argmax(params[3:7]) == 1
        end
    end
end
