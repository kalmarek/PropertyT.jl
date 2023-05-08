using LinearAlgebra
BLAS.set_num_threads(8)

ENV["OMP_NUM_THREADS"] = 1

using Groups
import Groups.MatrixGroups

include(joinpath(@__DIR__, "../test/optimizers.jl"))
using PropertyT

using PropertyT.SymbolicWedderburn
using PropertyT.PermutationGroups
using PropertyT.StarAlgebras

include(joinpath(@__DIR__, "argparse.jl"))
include(joinpath(@__DIR__, "utils.jl"))

const N = parsed_args["N"]
const HALFRADIUS = parsed_args["halfradius"]
const UPPER_BOUND = parsed_args["upper_bound"]

const GENUS = 2N

G = MatrixGroups.SymplecticGroup{GENUS}(Int8)

RG, S, sizes = @time PropertyT.group_algebra(G, halfradius = HALFRADIUS)

wd = let RG = RG, N = N
    G = StarAlgebras.object(RG)
    P = PermGroup(perm"(1,2)", Perm(circshift(1:N, -1)))
    Σ = Groups.Constructions.WreathProduct(PermGroup(perm"(1,2)"), P)
    # Σ = P
    act = PropertyT.action_by_conjugation(G, Σ)
    @info "Computing WedderburnDecomposition"

    wdfl = @time SymbolicWedderburn.WedderburnDecomposition(
        Float64,
        Σ,
        act,
        basis(RG),
        StarAlgebras.Basis{UInt16}(@view basis(RG)[1:sizes[HALFRADIUS]]),
    )
    @info wdfl
    wdfl
end

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
        max_iters = 20_000,
        accel = 50,
        alpha = 1.95,
    ),
    data = (elt = elt, unit = unit, halfradius = HALFRADIUS),
)
