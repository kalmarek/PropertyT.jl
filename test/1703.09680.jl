function check_positivity(elt, unit; upper_bound=Inf, halfradius=2, optimizer)
    @time sos_problem =
        PropertyT.sos_problem_primal(elt, unit, upper_bound=upper_bound)

    status, _ = PropertyT.solve(sos_problem, optimizer)
    P = JuMP.value.(sos_problem[:P])
    Q = real.(sqrt(P))
    certified, λ_cert = PropertyT.certify_solution(
        elt,
        unit,
        JuMP.objective_value(sos_problem),
        Q,
        halfradius=halfradius,
    )
    return status, certified, λ_cert
end

@testset "1703.09680 Examples" begin

    @testset "SL(2,Z)" begin
        N = 2
        G = MatrixGroups.SpecialLinearGroup{N}(Int8)
        RG, S, sizes = PropertyT.group_algebra(G, halfradius=2, twisted=true)

        Δ = let RG = RG, S = S
            RG(length(S)) - sum(RG(s) for s in S)
        end

        elt = Δ^2
        unit = Δ
        ub = 0.1

        status, certified, λ = check_positivity(
            elt,
            unit,
            upper_bound=ub,
            halfradius=2,
            optimizer=scs_optimizer(
                eps=1e-10,
                max_iters=5_000,
                accel=50,
                alpha=1.9,
            )
        )

        @test status == JuMP.ALMOST_OPTIMAL
        @test !certified
        @test λ < 0
    end

    @testset "SL(3,F₅)" begin
        N = 3
        G = MatrixGroups.SpecialLinearGroup{N}(SymbolicWedderburn.Characters.FiniteFields.GF{5})
        RG, S, sizes = PropertyT.group_algebra(G, halfradius=2, twisted=true)

        Δ = let RG = RG, S = S
            RG(length(S)) - sum(RG(s) for s in S)
        end

        elt = Δ^2
        unit = Δ
        ub = 1.01 # 1.5

        status, certified, λ = check_positivity(
            elt,
            unit,
            upper_bound=ub,
            halfradius=2,
            optimizer=scs_optimizer(
                eps=1e-10,
                max_iters=5_000,
                accel=50,
                alpha=1.9,
            )
        )

        @test status == JuMP.OPTIMAL
        @test certified
        @test λ > 1
    end

    @testset "SAut(F₂)" begin
        N = 2
        G = SpecialAutomorphismGroup(FreeGroup(N))
        RG, S, sizes = PropertyT.group_algebra(G, halfradius=2, twisted=true)

        Δ = let RG = RG, S = S
            RG(length(S)) - sum(RG(s) for s in S)
        end

        elt = Δ^2
        unit = Δ
        ub = 0.1

        status, certified, λ = check_positivity(
            elt,
            unit,
            upper_bound=0.1,
            halfradius=2,
            optimizer=scs_optimizer(
                eps=1e-10,
                max_iters=5_000,
                accel=50,
                alpha=1.9,
            )
        )

        @test status == JuMP.ALMOST_OPTIMAL
        @test λ < 0
        @test !certified
    end

    @testset "SL(3,Z) has (T)" begin
        n = 3

        SL = MatrixGroups.SpecialLinearGroup{n}(Int8)
        RSL, S, sizes = PropertyT.group_algebra(SL, halfradius=2, twisted=true)

        Δ = RSL(length(S)) - sum(RSL(s) for s in S)

        @testset "basic formulation" begin
            elt = Δ^2
            unit = Δ
            ub = 0.1

            opt_problem = PropertyT.sos_problem_primal(
                elt,
                unit,
                upper_bound=ub,
                augmented=false,
            )

            status, _ = PropertyT.solve(
                opt_problem,
                cosmo_optimizer(
                    eps=1e-10,
                    max_iters=10_000,
                    accel=0,
                    alpha=1.5,
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
                Q,
                halfradius=2,
                augmented=false,
            )

            @test certified
            @test isapprox(λ_cert, λ, rtol=1e-5)
        end

        @testset "augmented formulation" begin
            elt = Δ^2
            unit = Δ
            ub = 0.20001 # Inf

            opt_problem = PropertyT.sos_problem_primal(
                elt,
                unit,
                upper_bound=ub,
                augmented=true,
            )

            status, _ = PropertyT.solve(
                opt_problem,
                scs_optimizer(
                    eps=1e-10,
                    max_iters=10_000,
                    accel=-10,
                    alpha=1.5,
                ),
            )

            @test status == JuMP.OPTIMAL

            λ = JuMP.value(opt_problem[:λ])
            Q = real.(sqrt(JuMP.value.(opt_problem[:P])))

            certified, λ_cert = PropertyT.certify_solution(
                elt,
                unit,
                λ,
                Q,
                halfradius=2,
                augmented=true,
            )

            @test certified
            @test isapprox(λ_cert, λ, rtol=1e-5)
            @test λ_cert > 2 // 10
        end
    end
end
