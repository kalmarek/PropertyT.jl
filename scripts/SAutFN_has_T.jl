using LinearAlgebra
BLAS.set_num_threads(4)
ENV["OMP_NUM_THREADS"] = 4
include(joinpath(@__DIR__, "../test/optimizers.jl"))
using SCS_MKL_jll

using Groups
import Groups.MatrixGroups

using PropertyT

import PropertyT.SW as SW
using PropertyT.PG
using PropertyT.SA
include(joinpath(@__DIR__, "argparse.jl"))

const N = parsed_args["N"]
const HALFRADIUS = parsed_args["halfradius"]
const UPPER_BOUND = parsed_args["upper_bound"]

G = SpecialAutomorphismGroup(FreeGroup(N))
@info "Running Δ² - λ·Δ sum of squares decomposition for " G

@info "computing group algebra structure"
RG, S, sizes = @time PropertyT.group_algebra(G, halfradius = HALFRADIUS)

@info "computing WedderburnDecomposition"
wd = let RG = RG, N = N
    G = StarAlgebras.object(RG)
    P = PermGroup(perm"(1,2)", Perm(circshift(1:N, -1)))
    Σ = Groups.Constructions.WreathProduct(PermGroup(perm"(1,2)"), P)
    act = PropertyT.action_by_conjugation(G, Σ)

    wdfl = @time SW.WedderburnDecomposition(
        Float64,
        Σ,
        act,
        basis(RG),
        StarAlgebras.Basis{UInt16}(@view basis(RG)[1:sizes[HALFRADIUS]]),
    )
end
@info wd

Δ = RG(length(S)) - sum(RG(s) for s in S)
elt = Δ^2;
unit = Δ;
warm = nothing

@info "defining optimization problem"
@time model, varP = PropertyT.sos_problem_primal(
    elt,
    unit,
    wd;
    upper_bound = UPPER_BOUND,
    augmented = true,
    show_progress = true,
)

let status = JuMP.OPTIMIZE_NOT_CALLED, warm = warm, eps = 1e-10
    certified, λ = false, 0.0
    while status ≠ JuMP.OPTIMAL
        @time status, warm = PropertyT.solve(
            model,
            scs_optimizer(;
                linear_solver = SCS.MKLDirectSolver,
                eps = eps,
                max_iters = N * 10_000,
                accel = 50,
                alpha = 1.95,
            ),
            warm,
        )

        @info "reconstructing the solution"
        Q = @time let wd = wd, Ps = [JuMP.value.(P) for P in varP], eps = 1e-10
            PropertyT.__droptol!.(Ps, 100eps)
            Qs = real.(sqrt.(Ps))
            PropertyT.__droptol!.(Qs, eps)

            PropertyT.reconstruct(Qs, wd)
        end

        @info "certifying the solution"
        certified, λ = PropertyT.certify_solution(
            elt,
            unit,
            JuMP.objective_value(model),
            Q;
            halfradius = HALFRADIUS,
            augmented = true,
        )
    end

    if certified && λ > 0
        Κ(λ, S) = round(sqrt(2λ / length(S)), Base.RoundDown; digits = 5)
        @info "Certified result: $G has property (T):" N λ Κ(λ, S)
    else
        @info "Could NOT certify the result:" certified λ
    end
end
