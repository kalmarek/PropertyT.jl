using AbstractAlgebra, Nemo, Groups, SCS
using SparseArrays
using JLD
using PropertyT
using Test

indexing(n) = [(i,j) for i in 1:n for j in (i+1):n]
function Groups.gens(M::MatSpace)
    @assert M.cols == M.rows
    N = M.cols
    E(i,j) = begin g = M(1); g[i,j] = 1; g end
    S = [E(i,j) for (i,j) in indexing(N)]
    S = [S; transpose.(S)]
    return S
end

solver(iters; accel=1) = 
    SCSSolver(max_iters=iters, acceleration_lookback=accel, eps=1e-10)
    
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

        sett = PropertyT.Settings("SL($N,Z)", G, S, autS, solver(2000, accel=10);
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
        sett = PropertyT.Settings("SL($N,Z)", G, S, autS, solver(5000, accel=10);
        upper_bound=1.3, warmstart=false)
        
        @test PropertyT.check_property_T(sett) == false
        #second run just checks the obtained solution
        @test PropertyT.check_property_T(sett) == false 

        sett = PropertyT.Settings("SL($N,Z)", G, S, autS, solver(20000, accel=10);
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
        
        sett = PropertyT.Settings("SAut(F$N)", G, S, autS, solver(10000);
        upper_bound=0.15, warmstart=false)
        
        @test PropertyT.check_property_T(sett) == false
    end
end
