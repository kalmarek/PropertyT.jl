using SparseArrays

@testset "Sq, Adj, Op" begin
    function isconstant_on_orbit(v, orb)
        isempty(orb) && return true
        k = v[first(orb)]
        return all(v[o] == k for o in orb)
    end

    @testset "unit tests" begin
        @test PropertyT.isopposite(PG.perm"(1,2,3)(4)", PG.perm"(1,4,2)")
        @test PropertyT.isadjacent(PG.perm"(1,2,3)", PG.perm"(1,2)(3)")

        @test !PropertyT.isopposite(PG.perm"(1,2,3)", PG.perm"(1,2)(3)")
        @test !PropertyT.isadjacent(PG.perm"(1,4)", PG.perm"(2,3)(4)")

        @test isconstant_on_orbit([1, 1, 1, 2, 2], [2, 3])
        @test !isconstant_on_orbit([1, 1, 1, 2, 2], [2, 3, 4])
    end

    @testset "Sq, Adj, Op in SL(4,Z)" begin
        N = 4
        G = MatrixGroups.SpecialLinearGroup{N}(Int8)
        @info "running tests for" G

        RG, S, sizes = PropertyT.group_algebra(G; halfradius = 2)

        Δ = let RG = RG, S = S
            RG(length(S)) - sum(RG(s) for s in S)
        end

        P = PG.PermGroup(PG.perm"(1,2)", PG.Perm(circshift(1:N, -1)))
        Σ = Groups.Constructions.WreathProduct(PG.PermGroup(PG.perm"(1,2)"), P)
        act = PropertyT.action_by_conjugation(G, Σ)

        wd = SW.WedderburnDecomposition(
            Float64,
            Σ,
            act,
            SA.basis(RG),
            SA.Basis{UInt16}(@view SA.basis(RG)[1:sizes[2]]),
        )
        @info wd
        ivs = SW.invariant_vectors(wd)

        sq, adj, op = PropertyT.SqAdjOp(RG, N)

        @test all(
            isconstant_on_orbit(sq, SparseArrays.nonzeroinds(iv)) for iv in ivs
        )

        @test all(
            isconstant_on_orbit(adj, SparseArrays.nonzeroinds(iv)) for iv in ivs
        )

        @test all(
            isconstant_on_orbit(op, SparseArrays.nonzeroinds(iv)) for iv in ivs
        )

        e = one(G)
        g = G([alphabet(G)[MatrixGroups.ElementaryMatrix{N}(1, 2, Int8(1))]])
        h = G([alphabet(G)[MatrixGroups.ElementaryMatrix{N}(1, 3, Int8(1))]])
        k = G([alphabet(G)[MatrixGroups.ElementaryMatrix{N}(3, 4, Int8(1))]])

        @test sq[e] == 120
        @test sq[g] == sq[h] == -8
        @test sq[g^2] == sq[h^2] == 1
        @test sq[g*h] == sq[h*g] == 0

        @test adj[e] == 384
        @test adj[g] == adj[h] == -32
        @test adj[g^2] == adj[h^2] == 0
        @test adj[g*h] == adj[h*g] == 2
        @test adj[k*h] == adj[h*k] == 1

        @test op[e] == 96
        @test op[g] == op[h] == -8
        @test op[g^2] == op[h^2] == 0
        @test op[g*h] == op[h*g] == 0
        @test op[g*k] == op[k*g] == 2
        @test op[h*k] == op[k*h] == 0
    end

    @testset "SAut(F₃)" begin
        n = 3
        G = SpecialAutomorphismGroup(FreeGroup(n))
        RG, S, sizes = PropertyT.group_algebra(G; halfradius = 2)
        sq, adj, op = PropertyT.SqAdjOp(RG, n)

        @test sq(one(G)) == 216
        @test all(sq(g) == -16 for g in gens(G))
        @test adj(one(G)) == 384
        @test all(adj(g) == -32 for g in gens(G))
        @test iszero(op)
    end
end

@testset "1812.03456 examples" begin
    @testset "SL(3,Z)" begin
        n = 3

        G = MatrixGroups.SpecialLinearGroup{n}(Int8)
        @info "running tests for" G
        RG, S, sizes = PropertyT.group_algebra(G; halfradius = 2)

        Δ = RG(length(S)) - sum(RG(s) for s in S)

        P = PG.PermGroup(PG.perm"(1,2)", PG.Perm(circshift(1:n, -1)))
        Σ = Groups.Constructions.WreathProduct(PG.PermGroup(PG.perm"(1,2)"), P)
        act = PropertyT.action_by_conjugation(G, Σ)

        wd = SW.WedderburnDecomposition(
            Float64,
            Σ,
            act,
            SA.basis(RG),
            SA.Basis{UInt16}(@view SA.basis(RG)[1:sizes[2]]),
        )
        @info wd

        sq, adj, op = PropertyT.SqAdjOp(RG, n)

        @testset "Sq₃ is SOS" begin
            elt = sq
            UB = Inf # λ ≈ 0.1040844

            status, certified, λ_cert = check_positivity(
                elt,
                Δ,
                wd;
                upper_bound = UB,
                halfradius = 2,
                optimizer = scs_optimizer(; accel = -50, alpha = 1.9),
            )
            @test status == JuMP.OPTIMAL
            @test certified
            @test λ_cert > 104 // 1000
        end

        @testset "Adj₃ is SOS" begin
            elt = adj
            UB = Inf # λ ≈ 0.15858018

            status, certified, λ_cert = check_positivity(
                elt,
                Δ,
                wd;
                upper_bound = UB,
                halfradius = 2,
                optimizer = cosmo_optimizer(; accel = 50, alpha = 1.9),
            )
            @test status == JuMP.OPTIMAL
            @test certified
            @test λ_cert > 1585 // 10000

            m, _ = PropertyT.sos_problem_primal(elt, wd)
            PropertyT.solve(
                m,
                scs_optimizer(; max_iters = 1000, accel = 50, alpha = 1.9),
            )

            @test JuMP.termination_status(m) in
                  (JuMP.ALMOST_OPTIMAL, JuMP.OPTIMAL, JuMP.ITERATION_LIMIT)
            @test abs(JuMP.objective_value(m)) < 1e-3
        end

        @testset "Op₃ is empty, so can not be certified" begin
            elt = op
            @test iszero(op)

            UB = Inf

            status, certified, λ_cert = check_positivity(
                elt,
                Δ,
                wd;
                upper_bound = UB,
                halfradius = 2,
                optimizer = scs_optimizer(; accel = 50, alpha = 1.9),
            )
            @test status == JuMP.OPTIMAL
            @test !certified
            @test λ_cert < 0
        end
    end

    @testset "SL(4,Z)" begin
        n = 4

        G = MatrixGroups.SpecialLinearGroup{n}(Int8)
        @info "running tests for" G
        RG, S, sizes = PropertyT.group_algebra(G; halfradius = 2)

        Δ = RG(length(S)) - sum(RG(s) for s in S)

        P = PG.PermGroup(PG.perm"(1,2)", PG.Perm(circshift(1:n, -1)))
        Σ = Groups.Constructions.WreathProduct(PG.PermGroup(PG.perm"(1,2)"), P)
        act = PropertyT.action_by_conjugation(G, Σ)

        wd = SW.WedderburnDecomposition(
            Float64,
            Σ,
            act,
            SA.basis(RG),
            SA.Basis{UInt16}(@view SA.basis(RG)[1:sizes[2]]),
        )
        @info wd

        sq, adj, op = PropertyT.SqAdjOp(RG, n)

        @testset "Sq is SOS" begin
            elt = sq
            UB = Inf # λ ≈ 0.31670

            status, certified, λ_cert = check_positivity(
                elt,
                Δ,
                wd;
                upper_bound = UB,
                halfradius = 2,
                optimizer = cosmo_optimizer(; accel = 50, alpha = 1.9),
            )
            @test status == JuMP.OPTIMAL
            @test certified
            @test λ_cert > 316 // 1000
        end

        @testset "Adj is SOS" begin
            elt = adj
            UB = 0.541 # λ ≈ 0.545710

            status, certified, λ_cert = check_positivity(
                elt,
                Δ,
                wd;
                upper_bound = UB,
                halfradius = 2,
                optimizer = cosmo_optimizer(; accel = 50, alpha = 1.9),
            )
            @test status == JuMP.OPTIMAL
            @test certified
            @test λ_cert > 54 // 100
        end

        @testset "Op is a sum of squares, but not an order unit" begin
            elt = op
            UB = Inf

            status, certified, λ_cert = check_positivity(
                elt,
                Δ,
                wd;
                upper_bound = UB,
                halfradius = 2,
                optimizer = cosmo_optimizer(; accel = 50, alpha = 1.9),
            )
            @test status == JuMP.OPTIMAL
            @test !certified
            @test -1e-2 < λ_cert < 0
        end
    end
end
