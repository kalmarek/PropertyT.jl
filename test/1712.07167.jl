@testset "1712.07167 Examples" begin

    @testset "SAut(F₃)" begin
        N = 3
        G = SpecialAutomorphismGroup(FreeGroup(N))
        RG, S, sizes = PropertyT.group_algebra(G, halfradius=2, twisted=true)

        P = PermGroup(perm"(1,2)", Perm(circshift(1:N, -1)))
        Σ = PropertyT.Constructions.WreathProduct(PermGroup(perm"(1,2)"), P)
        act = PropertyT.action_by_conjugation(G, Σ)
        wd = WedderburnDecomposition(
            Float64,
            Σ,
            act,
            basis(RG),
            StarAlgebras.Basis{UInt16}(@view basis(RG)[1:sizes[2]]),
        )

        Δ = let RG = RG, S = S
            RG(length(S)) - sum(RG(s) for s in S)
        end

        elt = Δ^2
        unit = Δ
        ub = Inf

        status, certified, λ_cert = check_positivity(
            elt,
            unit,
            wd,
            upper_bound=ub,
            halfradius=2,
            optimizer=cosmo_optimizer(
                eps=1e-7,
                max_iters=10_000,
                accel=50,
                alpha=1.9,
            ),
        )

        @test status == JuMP.OPTIMAL
        @test !certified
        @test λ_cert < 0
    end

    @testset "SL(3,Z) has (T)" begin
        n = 3

        SL = MatrixGroups.SpecialLinearGroup{n}(Int8)
        RSL, S, sizes = PropertyT.group_algebra(SL, halfradius=2, twisted=true)

        Δ = RSL(length(S)) - sum(RSL(s) for s in S)

        @testset "Wedderburn formulation" begin
            P = PermGroup(perm"(1,2)", Perm(circshift(1:n, -1)))
            Σ = PropertyT.Constructions.WreathProduct(PermGroup(perm"(1,2)"), P)
            act = PropertyT.action_by_conjugation(SL, Σ)
            wd = WedderburnDecomposition(
                Rational{Int},
                Σ,
                act,
                basis(RSL),
                StarAlgebras.Basis{UInt16}(@view basis(RSL)[1:sizes[2]]),
            )

            elt = Δ^2
            unit = Δ
            ub = 0.2801

            @test_throws ErrorException PropertyT.sos_problem_primal(
                elt,
                unit,
                wd,
                upper_bound=ub,
                augmented=false,
            )

            wdfl = SymbolicWedderburn.WedderburnDecomposition(
                Float64,
                Σ,
                act,
                basis(RSL),
                StarAlgebras.Basis{UInt16}(@view basis(RSL)[1:sizes[2]]),
            )

            model, varP = PropertyT.sos_problem_primal(
                elt,
                unit,
                wdfl,
                upper_bound=ub,
                augmented=false,
            )

            status, warm = PropertyT.solve(
                model,
                cosmo_optimizer(
                    eps=1e-10,
                    max_iters=20_000,
                    accel=50,
                    alpha=1.9,
                ),
            )

            @test status == JuMP.OPTIMAL

            status, _ = PropertyT.solve(
                model,
                scs_optimizer(
                    eps=1e-10,
                    max_iters=100,
                    accel=-20,
                    alpha=1.2,
                ),
                warm
            )

            @test status == JuMP.OPTIMAL

            Q = @time let varP = varP
                Qs = map(varP) do P
                    real.(sqrt(JuMP.value.(P)))
                end
                PropertyT.reconstruct(Qs, wdfl)
            end
            λ = JuMP.value(model[:λ])

            sos = PropertyT.compute_sos(parent(elt), Q; augmented=false)

            certified, λ_cert = PropertyT.certify_solution(
                elt,
                unit,
                λ,
                Q,
                halfradius=2,
                augmented=false,
            )

            @test certified
            @test λ_cert >= 28 // 100
        end

        @testset "augmented Wedderburn formulation" begin
            elt = Δ^2
            unit = Δ
            ub = Inf

            P = PermGroup(perm"(1,2)", Perm(circshift(1:n, -1)))
            Σ = PropertyT.Constructions.WreathProduct(PermGroup(perm"(1,2)"), P)
            act = PropertyT.action_by_conjugation(SL, Σ)

            wdfl = SymbolicWedderburn.WedderburnDecomposition(
                Float64,
                Σ,
                act,
                basis(RSL),
                StarAlgebras.Basis{UInt16}(@view basis(RSL)[1:sizes[2]]),
            )

            opt_problem, varP = PropertyT.sos_problem_primal(
                elt,
                unit,
                wdfl,
                upper_bound=ub,
                # augmented = true # since both elt and unit are augmented
            )

            status, _ = PropertyT.solve(
                opt_problem,
                scs_optimizer(
                    eps=1e-8,
                    max_iters=20_000,
                    accel=0,
                    alpha=1.9,
                ),
            )

            @test status == JuMP.OPTIMAL

            Q = @time let varP = varP
                Qs = map(varP) do P
                    real.(sqrt(JuMP.value.(P)))
                end
                PropertyT.reconstruct(Qs, wdfl)
            end

            certified, λ_cert = PropertyT.certify_solution(
                elt,
                unit,
                JuMP.objective_value(opt_problem),
                Q,
                halfradius=2,
                # augmented = true # since both elt and unit are augmented
            )

            @test certified
            @test λ_cert > 28 // 100
        end
    end
end
