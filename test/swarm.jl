@testset "Swarm Type" begin
    @test_throws ArgumentError ParticleSwarm(2)
    ps = ParticleSwarm()
    @test_throws ArgumentError ps.n_particles = 2
end