@testset "1703.09680 Examples" begin

    @testset "SL(2,Z)" begin
        N = 2
        G = MatrixSpace(Nemo.ZZ, N,N)
        S = Groups.gens(G)
        S = [S; inv.(S)]

        rm("SL($N,Z)", recursive=true, force=true)
        sett = PropertyT.Settings("SL($N,Z)", G, S, solver(20000, accel=20); upper_bound=0.1)
        
        @test PropertyT.check_property_T(sett) == false
    end

    @testset "SL(3,Z)" begin
        N = 3
        G = MatrixSpace(Nemo.ZZ, N,N)
        S = Groups.gens(G)
        S = [S; inv.(S)]
        
        rm("SL($N,Z)", recursive=true, force=true)
        sett = PropertyT.Settings("SL($N,Z)", G, S, solver(1000, accel=20); upper_bound=0.1)
        
        @test PropertyT.check_property_T(sett) == true
    end
    
    @testset "SAut(F₂)" begin
        N = 2
        G = SAut(FreeGroup(N))
        S = Groups.gens(G)
        S = [S; inv.(S)]
        
        rm("SAut(F$N)", recursive=true, force=true)
        sett = PropertyT.Settings("SAut(F$N)", G, S, solver(20000);
        upper_bound=0.15, warmstart=false)
        
        @test PropertyT.check_property_T(sett) == false
    end
end
