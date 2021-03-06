@testset "actions on Group[Rings]" begin
    Eij = PropertyT.EltaryMat
    ssgs(M::MatAlgebra, i, j) = (S = [Eij(M, i, j), Eij(M, j, i)];
        S = unique([S; inv.(S)]); S)

    function ssgs(A::AutGroup, i, j)
        rmuls = [Groups.transvection_R(i,j), Groups.transvection_R(j,i)]
        lmuls = [Groups.transvection_L(i,j), Groups.transvection_L(j,i)]
        gen_set = A.([rmuls; lmuls])
        return unique([gen_set; inv.(gen_set)])
    end

@testset "actions on SL(3,Z) and its group ring" begin
    N = 3
    halfradius = 2
    M = MatrixAlgebra(zz, N)
    S = PropertyT.generating_set(M)
    E_R, sizes = Groups.wlmetric_ball(S, one(M), radius=2halfradius);

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

        for G in [SymmetricGroup(N), WreathProduct(SymmetricGroup(2), SymmetricGroup(N))]
            @test all(one(M)^g == one(M) for g in G)
            @test all(rdict[m^g] <= sizes[1] for g in G for m in S)
            @test all(m^g*n^g == (m*n)^g for g in G for m in S for n in S)

            @test all(Δ^g == Δ for g in G)
            @test all(x^g == RG(1) - RG(elt^g) for g in G)

            @test all(2RG(elt2^g) - RG(elt^g) == y^g for g in G)
        end
    end

    @testset "small Laplacians" begin
        for (i,j) in PropertyT.indexing(N)
            Sij = ssgs(M, i,j)
            Δij= PropertyT.spLaplacian(RG, Sij)

            @test all(Δij^p == PropertyT.spLaplacian(RG, ssgs(M, p[i], p[j])) for p in SymmetricGroup(N))

            @test all(Δij^g == PropertyT.spLaplacian(RG, ssgs(M, g.p[i], g.p[j])) for g in WreathProduct(SymmetricGroup(2), SymmetricGroup(N)))
        end
    end
end

@testset "actions on SAut(F_3) and its group ring" begin
    N = 3
    halfradius = 2
    M = SAut(FreeGroup(N))
    S = PropertyT.generating_set(M)
    E_R, sizes = Groups.wlmetric_ball(S, one(M), radius=2halfradius);

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

        for G in [SymmetricGroup(N), WreathProduct(SymmetricGroup(2), SymmetricGroup(N))]
            @test all(one(M)^g == one(M) for g in G)
            @test all(rdict[m^g] <= sizes[1] for g in G for m in S)
            @test all(m^g*n^g == (m*n)^g for g in G for m in S for n in S)

            @test all(Δ^g == Δ for g in G)
            @test all(x^g == RG(1) - RG(elt^g) for g in G)

            @test all(2RG(elt2^g) - RG(elt^g) == y^g for g in G)
        end
    end

    for (i,j) in PropertyT.indexing(N)
        Sij = ssgs(M, i,j)
        Δij= PropertyT.spLaplacian(RG, Sij)

        @test all(Δij^p == PropertyT.spLaplacian(RG, ssgs(M, p[i], p[j])) for p in SymmetricGroup(N))

        @test all(Δij^g == PropertyT.spLaplacian(RG, ssgs(M, g.p[i], g.p[j])) for g in WreathProduct(SymmetricGroup(2), SymmetricGroup(N)))
    end
end

end
