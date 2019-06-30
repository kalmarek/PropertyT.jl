@testset "1712.07167 Examples" begin

    @testset "oSL(3,Z)" begin
        N = 3
        G = MatrixAlgebra(zz, N)
        S = PropertyT.generating_set(G)
        autS = WreathProduct(PermGroup(2), PermGroup(N))

        rm("oSL($N,Z)", recursive=true, force=true)
        sett = PropertyT.Settings("SL($N,Z)", G, S, autS, with_SCS(2000, accel=20);
        upper_bound=0.27, warmstart=false)

        PropertyT.print_summary(sett)

        λ = PropertyT.spectral_gap(sett)
        @test λ < 0.0
        @test PropertyT.interpret_results(sett, λ) == false

        # second run just checks the solution due to warmstart=false above
        @test λ == PropertyT.spectral_gap(sett)
        @test PropertyT.check_property_T(sett) == false

        sett = PropertyT.Settings("SL($N,Z)", G, S, autS, with_SCS(2000, accel=20);
        upper_bound=0.27, warmstart=true)

        PropertyT.print_summary(sett)

        λ = PropertyT.spectral_gap(sett)
        @test λ > 0.269999
        @test PropertyT.interpret_results(sett, λ) == true

        # this should be very fast due to warmstarting:
        @test λ ≈ PropertyT.spectral_gap(sett) atol=1e-5
        @test PropertyT.check_property_T(sett) == true
    end

    @testset "oSL(4,Z)" begin
        N = 4
        G = MatrixAlgebra(zz, N)
        S = PropertyT.generating_set(G)
        autS = WreathProduct(PermGroup(2), PermGroup(N))

        rm("oSL($N,Z)", recursive=true, force=true)
        sett = PropertyT.Settings("SL($N,Z)", G, S, autS, with_SCS(2000, accel=20);
        upper_bound=1.3, warmstart=false)

        PropertyT.print_summary(sett)

        λ = PropertyT.spectral_gap(sett)
        @test λ < 0.0
        @test PropertyT.interpret_results(sett, λ) == false

        # second run just checks the solution due to warmstart=false above
        @test λ == PropertyT.spectral_gap(sett)
        @test PropertyT.check_property_T(sett) == false

        sett = PropertyT.Settings("SL($N,Z)", G, S, autS, with_SCS(5000, accel=20);
        upper_bound=1.3, warmstart=true)

        PropertyT.print_summary(sett)

        λ = PropertyT.spectral_gap(sett)
        @test λ > 1.2999
        @test PropertyT.interpret_results(sett, λ) == true

        # this should be very fast due to warmstarting:
        @test λ ≈ PropertyT.spectral_gap(sett) atol=1e-5
        @test PropertyT.check_property_T(sett) == true
    end

    @testset "SAut(F₃)" begin
        N = 3
        G = SAut(FreeGroup(N))
        S = PropertyT.generating_set(G)
        autS = WreathProduct(PermGroup(2), PermGroup(N))

        rm("oSAut(F$N)", recursive=true, force=true)

        sett = PropertyT.Settings("SAut(F$N)", G, S, autS, with_SCS(1000);
        upper_bound=0.15, warmstart=false)

        PropertyT.print_summary(sett)

        @test PropertyT.check_property_T(sett) == false
    end
end
