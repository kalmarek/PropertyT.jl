function check_positivity(elt, unit, wd; upper_bound=Inf, halfradius=2, optimizer)
    @assert aug(elt) == aug(unit) == 0
    @time sos_problem, Ps =
        PropertyT.sos_problem_primal(elt, unit, wd, upper_bound=upper_bound)

    @time status, _ = PropertyT.solve(sos_problem, optimizer)

    Q = let Ps = Ps
        flPs = [real.(sqrt(JuMP.value.(P))) for P in Ps]
        PropertyT.reconstruct(flPs, wd)
    end

    λ = JuMP.value(sos_problem[:λ])

    sos = let RG = parent(elt), Q = Q
        z = zeros(eltype(Q), length(basis(RG)))
        res = AlgebraElement(z, RG)
        cnstrs = PropertyT.constraints(basis(RG), RG.mstructure, augmented=true)
        PropertyT._cnstr_sos!(res, Q, cnstrs)
    end

    residual = elt - λ * unit - sos
    λ_fl = PropertyT.sufficient_λ(residual, λ, halfradius=2)

    λ_fl < 0 && return status, false, λ_fl

    sos = let RG = parent(elt), Q = [PropertyT.IntervalArithmetic.@interval(q) for q in Q]
        z = zeros(eltype(Q), length(basis(RG)))
        res = AlgebraElement(z, RG)
        cnstrs = PropertyT.constraints(basis(RG), RG.mstructure, augmented=true)
        PropertyT._cnstr_sos!(res, Q, cnstrs)
    end

    λ_int = PropertyT.IntervalArithmetic.@interval(λ)

    residual_int = elt - λ_int * unit - sos
    λ_int = PropertyT.sufficient_λ(residual_int, λ_int, halfradius=2)

    return status, λ_int > 0, PropertyT.IntervalArithmetic.inf(λ_int)
end

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
                scs_optimizer(
                    eps=1e-10,
                    max_iters=20_000,
                    accel=-20,
                    alpha=1.2,
                ),
            )

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
