@testset "1712.07167 Examples" begin

    @testset "oSL(3,Z)" begin
        N = 3
        G = MatrixSpace(Nemo.ZZ, N,N)
        S = Groups.gens(G)
        S = [S; inv.(S)]
        autS = WreathProduct(PermGroup(2), PermGroup(N))
        
        rm("oSL($N,Z)", recursive=true, force=true)
        sett = PropertyT.Settings("SL($N,Z)", G, S, autS, solver(2000, accel=20);
        upper_bound=0.27, warmstart=false)
       
        @test PropertyT.check_property_T(sett) == false
        #second run just checks the solution 
        @test PropertyT.check_property_T(sett) == false 

        sett = PropertyT.Settings("SL($N,Z)", G, S, autS, solver(2000, accel=20);
        upper_bound=0.27, warmstart=true)
        
        @test PropertyT.check_property_T(sett) == true
    end
    
    @testset "oSL(4,Z)" begin
        N = 4
        G = MatrixSpace(Nemo.ZZ, N,N)
        S = Groups.gens(G)
        S = [S; inv.(S)]
        autS = WreathProduct(PermGroup(2), PermGroup(N))
        
        rm("oSL($N,Z)", recursive=true, force=true)
        sett = PropertyT.Settings("SL($N,Z)", G, S, autS, solver(2000, accel=20);
        upper_bound=1.3, warmstart=false)
        
        @test PropertyT.check_property_T(sett) == false
        #second run just checks the obtained solution
        @test PropertyT.check_property_T(sett) == false 

        sett = PropertyT.Settings("SL($N,Z)", G, S, autS, solver(5000, accel=20);
        upper_bound=1.3, warmstart=true)
        
        @test PropertyT.check_property_T(sett) == true
    end

    @testset "SAut(F₃)" begin
        N = 3
        G = SAut(FreeGroup(N))
        S = Groups.gens(G)
        S = [S; inv.(S)]
        autS = WreathProduct(PermGroup(2), PermGroup(N))
        
        rm("oSAut(F$N)", recursive=true, force=true)
        
        sett = PropertyT.Settings("SAut(F$N)", G, S, autS, solver(5000);
        upper_bound=0.15, warmstart=false)
        
        @test PropertyT.check_property_T(sett) == false
    end
end
