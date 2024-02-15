@testset "1703.09680 Examples" begin
    @testset "SL(2,Z)" begin
        N = 2
        G = MatrixGroups.SpecialLinearGroup{N}(Int8)
        RG, S, sizes = PropertyT.group_algebra(G; halfradius = 2)

        Δ = let RG = RG, S = S
            RG(length(S)) - sum(RG(s) for s in S)
        end

        elt = Δ^2
        unit = Δ
        ub = 0.1

        status, certified, λ = check_positivity(
            elt,
            unit;
            upper_bound = ub,
            halfradius = 2,
            optimizer = scs_optimizer(;
                eps = 1e-10,
                max_iters = 5_000,
                accel = 50,
                alpha = 1.9,
            ),
        )

        @test status == JuMP.ALMOST_OPTIMAL
        @test !certified
        @test λ < 0
    end

    @testset "SL(3,F₅)" begin
        N = 3
        G = MatrixGroups.SpecialLinearGroup{N}(
            SW.Characters.FiniteFields.GF{5},
        )
        RG, S, sizes = PropertyT.group_algebra(G; halfradius = 2)

        Δ = let RG = RG, S = S
            RG(length(S)) - sum(RG(s) for s in S)
        end

        elt = Δ^2
        unit = Δ
        ub = 1.01 # 1.5

        status, certified, λ = check_positivity(
            elt,
            unit;
            upper_bound = ub,
            halfradius = 2,
            optimizer = scs_optimizer(;
                eps = 1e-10,
                max_iters = 5_000,
                accel = 50,
                alpha = 1.9,
            ),
        )

        @test status == JuMP.OPTIMAL
        @test certified
        @test λ > 1

        m = PropertyT.sos_problem_dual(elt, unit)
        PropertyT.solve(
            m,
            cosmo_optimizer(;
                eps = 1e-6,
                max_iters = 5_000,
                accel = 50,
                alpha = 1.9,
            ),
        )

        @test JuMP.termination_status(m) in (JuMP.ALMOST_OPTIMAL, JuMP.OPTIMAL)
        @test JuMP.objective_value(m) ≈ 1.5 atol = 1e-2
    end

    @testset "SAut(F₂)" begin
        N = 2
        G = SpecialAutomorphismGroup(FreeGroup(N))
        RG, S, sizes = PropertyT.group_algebra(G; halfradius = 2)

        Δ = let RG = RG, S = S
            RG(length(S)) - sum(RG(s) for s in S)
        end

        elt = Δ^2
        unit = Δ
        ub = 0.1

        status, certified, λ = check_positivity(
            elt,
            unit;
            upper_bound = ub,
            halfradius = 2,
            optimizer = scs_optimizer(;
                eps = 1e-10,
                max_iters = 5_000,
                accel = 50,
                alpha = 1.9,
            ),
        )

        @test status == JuMP.ALMOST_OPTIMAL
        @test λ < 0
        @test !certified

        @time sos_problem = PropertyT.sos_problem_primal(elt; upper_bound = ub)

        status, _ = PropertyT.solve(
            sos_problem,
            cosmo_optimizer(;
                eps = 1e-7,
                max_iters = 10_000,
                accel = 0,
                alpha = 1.9,
            ),
        )
        @test status == JuMP.OPTIMAL
        P = JuMP.value.(sos_problem[:P])
        Q = real.(sqrt(P))
        certified, λ_cert =
            PropertyT.certify_solution(elt, zero(elt), 0.0, Q; halfradius = 2)
        @test !certified
        @test λ_cert < 0
    end

    @testset "SL(3,Z) has (T)" begin
        n = 3

        SL = MatrixGroups.SpecialLinearGroup{n}(Int8)
        RSL, S, sizes = PropertyT.group_algebra(SL; halfradius = 2)

        Δ = RSL(length(S)) - sum(RSL(s) for s in S)

        @testset "basic formulation" begin
            elt = Δ^2
            unit = Δ
            ub = 0.1

            opt_problem = PropertyT.sos_problem_primal(
                elt,
                unit;
                upper_bound = ub,
                augmented = false,
            )

            status, _ = PropertyT.solve(
                opt_problem,
                cosmo_optimizer(;
                    eps = 1e-10,
                    max_iters = 10_000,
                    accel = 0,
                    alpha = 1.5,
                ),
            )

            @test status == JuMP.OPTIMAL

            λ = JuMP.value(opt_problem[:λ])
            @test λ > 0.09
            Q = real.(sqrt(JuMP.value.(opt_problem[:P])))

            certified, λ_cert = PropertyT.certify_solution(
                elt,
                unit,
                λ,
                Q;
                halfradius = 2,
                augmented = false,
            )

            @test certified
            @test isapprox(λ_cert, λ, rtol = 1e-5)
        end

        @testset "augmented formulation" begin
            elt = Δ^2
            unit = Δ
            ub = 0.20001 # Inf

            opt_problem = PropertyT.sos_problem_primal(
                elt,
                unit;
                upper_bound = ub,
                augmented = true,
            )

            status, _ = PropertyT.solve(
                opt_problem,
                scs_optimizer(;
                    eps = 1e-10,
                    max_iters = 10_000,
                    accel = -10,
                    alpha = 1.5,
                ),
            )

            @test status == JuMP.OPTIMAL

            λ = JuMP.value(opt_problem[:λ])
            Q = real.(sqrt(JuMP.value.(opt_problem[:P])))

            certified, λ_cert = PropertyT.certify_solution(
                elt,
                unit,
                λ,
                Q;
                halfradius = 2,
                augmented = true,
            )

            @test certified
            @test isapprox(λ_cert, λ, rtol = 1e-5)
            @test λ_cert > 2 // 10
        end
    end
end
