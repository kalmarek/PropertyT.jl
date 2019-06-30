@testset "Sq, Adj, Op" begin
    function isconstant_on_orbit(v, orb)
        isempty(orb) && return true
        k = v[first(orb)]
        return all(v[o] == k for o in orb)
    end

    @testset "unit tests" begin
        for N in [3,4]
            M = MatrixSpace(Nemo.ZZ, N,N)
            A = SAut(FreeGroup(N))
            @test length(PropertyT.generating_set(M)) == 2N*(N-1)
            S = PropertyT.generating_set(M)
            @test all(inv(s) ∈ S for s in S)
            @test length(PropertyT.generating_set(A)) == 4N*(N-1)
            S = PropertyT.generating_set(A)
            @test all(inv(s) ∈ S for s in S)
        end

        N = 4
        M = MatrixSpace(Nemo.ZZ, N,N)
        S = PropertyT.generating_set(M)

        @test PropertyT.E(M, 1, 2) isa MatElem
        e12 = PropertyT.E(M, 1, 2)
        @test e12[1,2] == 1
        @test inv(e12)[1,2] == -1
        @test e12 ∈ S

        @test PropertyT.isopposite(perm"(1,2,3)(4)", perm"(1,4,2)")
        @test PropertyT.isadjacent(perm"(1,2,3)", perm"(1,2)(3)")

        @test !PropertyT.isopposite(perm"(1,2,3)", perm"(1,2)(3)")
        @test !PropertyT.isadjacent(perm"(1,4)", perm"(2,3)(4)")

        @test isconstant_on_orbit([1,1,1,2,2], [2,3])
        @test !isconstant_on_orbit([1,1,1,2,2], [2,3,4])
    end

    @testset "Sq, Adj, Op" begin

        N = 4
        M = MatrixSpace(Nemo.ZZ, N,N)
        S = PropertyT.generating_set(M)
        Δ = PropertyT.Laplacian(S, 2)
        RG = parent(Δ)

        autS = WreathProduct(PermGroup(2), PermGroup(N))
        orbits = PropertyT.orbit_decomposition(autS, RG.basis)

        @test PropertyT.Sq(RG) isa GroupRingElem
        sq = PropertyT.Sq(RG)
        @test all(isconstant_on_orbit(sq, orb) for orb in orbits)

        @test PropertyT.Adj(RG) isa GroupRingElem
        adj = PropertyT.Adj(RG)
        @test all(isconstant_on_orbit(adj, orb) for orb in orbits)

        @test PropertyT.Op(RG) isa GroupRingElem
        op = PropertyT.Op(RG)
        @test all(isconstant_on_orbit(op, orb) for orb in orbits)

        sq, adj, op = PropertyT.SqAdjOp(RG, N)
        @test sq == PropertyT.Sq(RG)
        @test adj == PropertyT.Adj(RG)
        @test op == PropertyT.Op(RG)

        e = one(M)
        g = PropertyT.E(M, 1,2)
        h = PropertyT.E(M, 1,3)
        k = PropertyT.E(M, 3,4)

        edges = N*(N-1)÷2
        @test sq[e] == 20*edges
        @test sq[g] == sq[h] == -8
        @test sq[g^2] == sq[h^2] == 1
        @test sq[g*h] == sq[h*g] == 0

        #     @test adj[e] == ...
        @test adj[g] == adj[h] # == ...
        @test adj[g^2] == adj[h^2] == 0
        @test adj[g*h] == adj[h*g] # == ...


        #     @test op[e] == ...
        @test op[g] == op[h] # == ...
        @test op[g^2] == op[h^2] == 0
        @test op[g*h] == op[h*g] == 0
        @test op[g*k] == op[k*g] # == ...
        @test op[h*k] == op[k*h] == 0
    end
end

