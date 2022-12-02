@testset "Quick tests" begin

    @testset "SL(2,F₇)" begin
        N = 2
        p = 7
        halfradius = 3
        G = MatrixGroups.SpecialLinearGroup{N}(SymbolicWedderburn.Characters.FiniteFields.GF{p})
        RG, S, sizes = PropertyT.group_algebra(G, halfradius=3, twisted=true)

        Δ = let RG = RG, S = S
            RG(length(S)) - sum(RG(s) for s in S)
        end

        elt = Δ^2
        unit = Δ
        ub = 0.58578# Inf# 1.5

        @testset "standard formulation" begin
            status, certified, λ_cert = check_positivity(
                elt,
                unit,
                upper_bound=ub,
                halfradius=2,
                optimizer=cosmo_optimizer(
                    eps=1e-7,
                    max_iters=5_000,
                    accel=50,
                    alpha=1.95,
                )
            )

            @test status == JuMP.OPTIMAL
            @test certified
            @test λ_cert > 5857 // 10000

            m = PropertyT.sos_problem_dual(elt, unit)
            PropertyT.solve(m, cosmo_optimizer(
                eps=1e-7,
                max_iters=10_000,
                accel=50,
                alpha=1.95,
            ))

            @test JuMP.termination_status(m) in (JuMP.ALMOST_OPTIMAL, JuMP.OPTIMAL)
            @test JuMP.objective_value(m) ≈ λ_cert atol = 1e-2
        end

        @testset "Wedderburn decomposition" begin
            P = PermGroup(perm"(1,2)", Perm(circshift(1:N, -1)))
            Σ = PropertyT.Constructions.WreathProduct(PermGroup(perm"(1,2)"), P)
            act = PropertyT.action_by_conjugation(G, Σ)

            wd = WedderburnDecomposition(
                Float64,
                Σ,
                act,
                basis(RG),
                StarAlgebras.Basis{UInt16}(@view basis(RG)[1:sizes[halfradius]]),
            )

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
            @test certified
            @test λ_cert > 5857 // 10000
        end
    end
end
