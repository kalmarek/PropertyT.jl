@testset "actions on Group[Rings]" begin
    Eij = PropertyT.EltaryMat
    ssgs(M::MatAlgebra, i, j) = (S = [Eij(M, i, j), Eij(M, j, i)];
        S = unique([S; inv.(S)]); S)

    rmul = Groups.rmul_autsymbol
    lmul = Groups.lmul_autsymbol

    function ssgs(A::AutGroup, i, j)
        rmuls = [rmul(i,j), rmul(j,i)]
        lmuls = [lmul(i,j), lmul(j,i)]
        gen_set = A.([rmuls; lmuls])
        return unique([gen_set; inv.(gen_set)])
    end

@testset "actions on SL(3,Z) and its group ring" begin
    N = 3
    halfradius = 2
    M = MatrixAlgebra(zz, N)
    S = PropertyT.generating_set(M)
    E_R, sizes = Groups.generate_balls(S, one(M), radius=2halfradius);

    rdict = GroupRings.reverse_dict(E_R)
    pm = GroupRings.create_pm(E_R, rdict, sizes[halfradius]; twisted=false);
    RG = GroupRing(M, E_R, rdict, pm)

    @testset "correctness of actions" begin
        Δ = length(S)*RG(1) - sum(RG(s) for s in S)
        @test Δ == PropertyT.spLaplacian(RG, S)

        elt = S[5]
        x = RG(1) - RG(elt)
        elt2 = E_R[rand(sizes[1]:sizes[2])]
        y = 2RG(elt2) - RG(elt)

        for G in [PermGroup(N), WreathProduct(PermGroup(2), PermGroup(N))]
            @test all(g(one(M)) == one(M) for g in G)
            @test all(rdict[g(m)] <= sizes[1] for g in G for m in S)
            @test all(g(m)*g(n) == g(m*n) for g in G for m in S for n in S)

            @test all(g(Δ) == Δ for g in G)
            @test all(g(x) == RG(1) - RG(g(elt)) for g in G)

            @test all(2RG(g(elt2)) - RG(g(elt)) == g(y) for g in G)
        end
    end

    @testset "small Laplacians" begin
        for (i,j) in PropertyT.indexing(N)
            Sij = ssgs(M, i,j)
            Δij= PropertyT.spLaplacian(RG, Sij)

            @test all(p(Δij) == PropertyT.spLaplacian(RG, ssgs(M, p[i], p[j])) for p in PermGroup(N))

            @test all(g(Δij) == PropertyT.spLaplacian(RG, ssgs(M, g.p[i], g.p[j])) for g in WreathProduct(PermGroup(2), PermGroup(N)))
        end
    end
end

@testset "actions on SAut(F_3) and its group ring" begin
    N = 3
    halfradius = 2
    M = SAut(FreeGroup(N))
    S = PropertyT.generating_set(M)
    E_R, sizes = Groups.generate_balls(S, one(M), radius=2halfradius);

    rdict = GroupRings.reverse_dict(E_R)
    pm = GroupRings.create_pm(E_R, rdict, sizes[halfradius]; twisted=false);
    RG = GroupRing(M, E_R, rdict, pm)


    @testset "correctness of actions" begin

        Δ = length(S)*RG(1) - sum(RG(s) for s in S)
        @test Δ == PropertyT.spLaplacian(RG, S)

        elt = S[5]
        x = RG(1) - RG(elt)
        elt2 = E_R[rand(sizes[1]:sizes[2])]
        y = 2RG(elt2) - RG(elt)

        for G in [PermGroup(N), WreathProduct(PermGroup(2), PermGroup(N))]
            @test all(g(one(M)) == one(M) for g in G)
            @test all(rdict[g(m)] <= sizes[1] for g in G for m in S)
            @test all(g(m)*g(n) == g(m*n) for g in G for m in S for n in S)

            @test all(g(Δ) == Δ for g in G)
            @test all(g(x) == RG(1) - RG(g(elt)) for g in G)

            @test all(2RG(g(elt2)) - RG(g(elt)) == g(y) for g in G)
        end
    end

    for (i,j) in PropertyT.indexing(N)
        Sij = ssgs(M, i,j)
        Δij= PropertyT.spLaplacian(RG, Sij)

        @test all(p(Δij) == PropertyT.spLaplacian(RG, ssgs(M, p[i], p[j])) for p in PermGroup(N))

        @test all(g(Δij) == PropertyT.spLaplacian(RG, ssgs(M, g.p[i], g.p[j])) for g in WreathProduct(PermGroup(2), PermGroup(N)))
    end
end

end
