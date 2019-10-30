@testset "1703.09680 Examples" begin

    @testset "SL(2,Z)" begin
        N = 2
        G = MatrixAlgebra(zz, N)
        S = PropertyT.generating_set(G)

        rm("SL($N,Z)", recursive=true, force=true)
        sett = PropertyT.Settings("SL($N,Z)", G, S, with_SCS(20000, accel=20); upper_bound=0.1)

        @info sett

        λ = PropertyT.spectral_gap(sett)
        @test λ < 0.0
        @test PropertyT.interpret_results(sett, λ) == false
    end

    @testset "SL(3,Z)" begin
        N = 3
        G = MatrixAlgebra(zz, N)
        S = PropertyT.generating_set(G)

        rm("SL($N,Z)", recursive=true, force=true)
        sett = PropertyT.Settings("SL($N,Z)", G, S, with_SCS(1000, accel=20); upper_bound=0.1)

        @info sett

        λ = PropertyT.spectral_gap(sett)
        @test λ > 0.0999
        @test PropertyT.interpret_results(sett, λ) == true

        @test PropertyT.check_property_T(sett) == true #second run should be fast
    end

    @testset "SAut(F₂)" begin
        N = 2
        G = SAut(FreeGroup(N))
        S = PropertyT.generating_set(G)

        rm("SAut(F$N)", recursive=true, force=true)
        sett = PropertyT.Settings("SAut(F$N)", G, S, with_SCS(10000);
        upper_bound=0.15)

        @info sett

        λ = PropertyT.spectral_gap(sett)
        @test λ < 0.0
        @test PropertyT.interpret_results(sett, λ) == false

    end
end
