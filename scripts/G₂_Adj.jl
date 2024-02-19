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

# const N = parsed_args["N"]
const HALFRADIUS = parsed_args["halfradius"]
const UPPER_BOUND = parsed_args["upper_bound"]

include(joinpath(@__DIR__, "./G₂_gens.jl"))

G, roots, Weyl = G₂_roots_weyl()
@info "Running Adj² - λ·Δ sum of squares decomposition for G₂"

@info "computing group algebra structure"
RG, S, sizes = @time PropertyT.group_algebra(G, halfradius = HALFRADIUS)

@info "computing WedderburnDecomposition"
wd = let Σ = Weyl, RG = RG
    act = PropertyT.AlphabetPermutation{eltype(Σ),Int64}(
        Dict(g => PermutationGroups.AP.perm(g) for g in Σ),
    )

    @time SW.WedderburnDecomposition(
        Float64,
        Σ,
        act,
        basis(RG),
        StarAlgebras.Basis{UInt16}(@view basis(RG)[1:sizes[HALFRADIUS]]),
        semisimple = false,
    )
end
@info wd

function desubscriptify(symbol::Symbol)
    digits = [
        Int(l) - 0x2080 for
        l in reverse(string(symbol)) if 0 ≤ Int(l) - 0x2080 ≤ 9
    ]
    res = 0
    for (i, d) in enumerate(digits)
        res += 10^(i - 1) * d
    end
    return res
end

function PropertyT.grading(g::MatrixGroups.MatrixElt, roots = roots)
    id = desubscriptify(g.id)
    return roots[id]
end

Δ = RG(length(S)) - sum(RG(s) for s in S)
Δs = PropertyT.laplacians(
    RG,
    S,
    x -> (gx = PropertyT.grading(x); Set([gx, -gx])),
)

elt = PropertyT.Adj(Δs)
@assert elt == Δ^2 - PropertyT.Sq(Δs)
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
    logdir = "./log/G2/r=$HALFRADIUS/Adj-$(UPPER_BOUND)Δ",
    optimizer = scs_optimizer(;
        linear_solver = SCS.MKLDirectSolver,
        eps = 1e-9,
        max_iters = 100_000,
        accel = 50,
        alpha = 1.95,
    ),
    data = (elt = elt, unit = unit, halfradius = HALFRADIUS),
)
