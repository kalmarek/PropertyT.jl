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
include(joinpath(@__DIR__, "utils.jl"))

const N = parsed_args["N"]
const HALFRADIUS = parsed_args["halfradius"]
const UPPER_BOUND = parsed_args["upper_bound"]

G = MatrixGroups.SymplecticGroup{2N}(Int8)
@info "Running Adj_C₂ - λ·Δ sum of squares decomposition for " G

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
    wdfl
end
@info wd

Δ = RG(length(S)) - sum(RG(s) for s in S)
Δs = PropertyT.laplacians(
    RG,
    S,
    x -> (gx = PropertyT.grading(x); Set([gx, -gx])),
)

# elt = Δ^2
elt = PropertyT.Adj(Δs, :C₂)
unit = Δ

@time model, varP = PropertyT.sos_problem_primal(
    elt,
    unit,
    wd;
    upper_bound = UPPER_BOUND,
    augmented = true,
    show_progress = true,
)

solve_in_loop(
    model,
    wd,
    varP;
    logdir = "./log/Sp($N,Z)/r=$HALFRADIUS/Adj_C₂-$(UPPER_BOUND)Δ",
    optimizer = cosmo_optimizer(;
        eps = 1e-10,
        max_iters = 50_000,
        accel = 50,
        alpha = 1.95,
    ),
    data = (elt = elt, unit = unit, halfradius = HALFRADIUS),
)
