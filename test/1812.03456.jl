@testset "Sq, Adj, Op" begin
    function isconstant_on_orbit(v, orb)
        isempty(orb) && return true
        k = v[first(orb)]
        return all(v[o] == k for o in orb)
    end

    @testset "unit tests" begin
        for N in [3,4]
            M = MatrixAlgebra(zz, N)

            @test PropertyT.EltaryMat(M, 1, 2) isa MatAlgElem
            e12 = PropertyT.EltaryMat(M, 1, 2)
            @test e12[1,2] == 1
            @test inv(e12)[1,2] == -1

            S = PropertyT.generating_set(M)
            @test e12 ∈ S

            @test length(PropertyT.generating_set(M)) == 2N*(N-1)
            @test all(inv(s) ∈ S for s in S)

            A = SAut(FreeGroup(N))
            @test length(PropertyT.generating_set(A)) == 4N*(N-1)
            S = PropertyT.generating_set(A)
            @test all(inv(s) ∈ S for s in S)
        end

        @test PropertyT.isopposite(perm"(1,2,3)(4)", perm"(1,4,2)")
        @test PropertyT.isadjacent(perm"(1,2,3)", perm"(1,2)(3)")

        @test !PropertyT.isopposite(perm"(1,2,3)", perm"(1,2)(3)")
        @test !PropertyT.isadjacent(perm"(1,4)", perm"(2,3)(4)")

        @test isconstant_on_orbit([1,1,1,2,2], [2,3])
        @test !isconstant_on_orbit([1,1,1,2,2], [2,3,4])
    end

    @testset "Sq, Adj, Op" begin

        N = 4
        M = MatrixAlgebra(zz, N)
        S = PropertyT.generating_set(M)
        Δ = PropertyT.Laplacian(S, 2)
        RG = parent(Δ)

        autS = WreathProduct(SymmetricGroup(2), SymmetricGroup(N))
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
        g = PropertyT.EltaryMat(M, 1,2)
        h = PropertyT.EltaryMat(M, 1,3)
        k = PropertyT.EltaryMat(M, 3,4)

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


@testset "1812.03456 examples" begin

    function SOS_residual(x::GroupRingElem, Q::Matrix)
        RG = parent(x)
        @time sos = PropertyT.compute_SOS(RG, Q);
        return x - sos
    end

    function check_positivity(elt, Δ, orbit_data, upper_bound, warm=nothing; with_solver=with_SCS(20_000, accel=10))
        SDP_problem, varP = PropertyT.SOS_problem_primal(elt, Δ, orbit_data; upper_bound=upper_bound)

        status, warm = PropertyT.solve(SDP_problem, with_solver, warm);
        Base.Libc.flush_cstdio()
        @info "Optimization status:" status

        λ = value(SDP_problem[:λ])
        Ps = [value.(P) for P in varP]

        Qs = real.(sqrt.(Ps));
        Q = PropertyT.reconstruct(Qs, orbit_data);

        b = SOS_residual(elt - λ*Δ, Q)
        return b, λ, warm
    end

    @testset "SL(3,Z)" begin
        N = 3
        halfradius = 2
        M = MatrixAlgebra(zz, N)
        S = PropertyT.generating_set(M)
        Δ = PropertyT.Laplacian(S, halfradius)
        RG = parent(Δ)
        orbit_data = PropertyT.BlockDecomposition(RG, WreathProduct(SymmetricGroup(2), SymmetricGroup(N)))
        orbit_data = PropertyT.decimate(orbit_data);

        @testset "Sq₃ is SOS" begin
            elt = PropertyT.Sq(RG)
            UB = 0.05 # 0.105?

            residual, λ, _ = check_positivity(elt, Δ, orbit_data, UB)
            Base.Libc.flush_cstdio()
            @info "obtained λ and residual" λ norm(residual, 1)

            @test 2^2*norm(residual, 1) < λ # i.e. we can certify positivity

            @test 2^2*norm(residual, 1) < 2λ/100
        end

        @testset "Adj₃ is SOS" begin
            elt = PropertyT.Adj(RG)
            UB = 0.1 # 0.157?

            residual, λ, _ = check_positivity(elt, Δ, orbit_data, UB)
            Base.Libc.flush_cstdio()
            @info "obtained λ and residual" λ norm(residual, 1)

            @test 2^2*norm(residual, 1) < λ
            @test 2^2*norm(residual, 1) < λ/100
        end

        @testset "Op₃ is empty, so can not be certified" begin
            elt = PropertyT.Op(RG)
            UB = Inf

            residual, λ, _ = check_positivity(elt, Δ, orbit_data, UB)
            Base.Libc.flush_cstdio()
            @info "obtained λ and residual" λ norm(residual, 1)

            @test 2^2*norm(residual, 1) > λ
        end
    end

    @testset "SL(4,Z)" begin
        N = 4
        halfradius = 2
        M = MatrixAlgebra(zz, N)
        S = PropertyT.generating_set(M)
        Δ = PropertyT.Laplacian(S, halfradius)
        RG = parent(Δ)
        orbit_data = PropertyT.BlockDecomposition(RG, WreathProduct(SymmetricGroup(2), SymmetricGroup(N)))
        orbit_data = PropertyT.decimate(orbit_data);

        @testset "Sq₄ is SOS" begin
            elt = PropertyT.Sq(RG)
            UB = 0.2 # 0.3172

            residual, λ, _ = check_positivity(elt, Δ, orbit_data, UB)
            Base.Libc.flush_cstdio()
            @info "obtained λ and residual" λ norm(residual, 1)

            @test 2^2*norm(residual, 1) < λ # i.e. we can certify positivity
            @test 2^2*norm(residual, 1) < λ/100
        end

        @testset "Adj₄ is SOS" begin
            elt = PropertyT.Adj(RG)
            UB = 0.3 # 0.5459?

            residual, λ, _ = check_positivity(elt, Δ, orbit_data, UB)
            @info "obtained λ and residual" λ norm(residual, 1)

            @test 2^2*norm(residual, 1) < λ # i.e. we can certify positivity
            @test 2^2*norm(residual, 1) < λ/100
        end

        @testset "we can't cerify that Op₄ SOS" begin
            elt = PropertyT.Op(RG)
            UB = 2.0

            residual, λ, _ = check_positivity(elt, Δ, orbit_data, UB,
            with_solver=with_SCS(20_000, accel=10, eps=2e-10))
            Base.Libc.flush_cstdio()
            @info "obtained λ and residual" λ norm(residual, 1)

            @test 2^2*norm(residual, 1) > λ # i.e. we can't certify positivity
        end

        @testset "Adj₄ + Op₄ is SOS" begin
            elt = PropertyT.Adj(RG) + PropertyT.Op(RG)
            UB = 0.6 # 0.82005

            residual, λ, _ = check_positivity(elt, Δ, orbit_data, UB)
            Base.Libc.flush_cstdio()
            @info "obtained λ and residual" λ norm(residual, 1)

            @test 2^2*norm(residual, 1) < λ # i.e. we can certify positivity
            @test 2^2*norm(residual, 1) < λ/100
        end
    end

    # @testset "Adj₄ + 100 Op₄ ∈ ISAut(F₄) is SOS" begin
    #     N = 4
    #     halfradius = 2
    #     M = SAut(FreeGroup(N))
    #     S = PropertyT.generating_set(M)
    #     Δ = PropertyT.Laplacian(S, halfradius)
    #     RG = parent(Δ)
    #     orbit_data = PropertyT.BlockDecomposition(RG, WreathProduct(SymmetricGroup(2), SymmetricGroup(N)))
    #     orbit_data = PropertyT.decimate(orbit_data);
    #
    #     @time elt = PropertyT.Adj(RG) + 100PropertyT.Op(RG)
    #     UB = 0.05
    #
    #     warm = nothing
    #
    #     residual, λ, warm = check_positivity(elt, Δ, orbit_data, UB, warm, with_solver=with_SCS(20_000, accel=10))
    #     @info "obtained λ and residual" λ norm(residual, 1)
    #
    #     @test 2^2*norm(residual, 1) < λ # i.e. we can certify positivity
    #     @test 2^2*norm(residual, 1) < λ/100
    # end
end
