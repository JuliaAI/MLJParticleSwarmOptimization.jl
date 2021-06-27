@testset "Swarm Options" begin
    @testset "Coefficients" begin
        @test_throws ArgumentError StaticCoeffs(w=-1.0)
        @test_throws ArgumentError StaticCoeffs(c1=-2.0)
        @test_throws ArgumentError StaticCoeffs(c2=-2.0)
        sc = StaticCoeffs(1, 2.0, 2.0f0)
        @test sc.w === 1.0
        @test sc.c1 === 2.0
        @test sc.c2 === 2.0
        @test_throws ArgumentError sc.w = -1.0
        @test_throws ArgumentError sc.c1 = -2.0
        @test_throws ArgumentError sc.c2 = -2.0
        sc = StaticCoeffs()
        @test PSO.coefficients(sc) == (1.0, 2.0, 2.0)
    end
end