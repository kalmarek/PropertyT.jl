using LinearAlgebra
BLAS.set_num_threads(1)
ENV["OMP_NUM_THREADS"] = 4

using MKL_jll
include(joinpath(@__DIR__, "../test/optimizers.jl"))

using Groups
import Groups.MatrixGroups
using PropertyT

using SymbolicWedderburn
using SymbolicWedderburn.StarAlgebras
using PermutationGroups

include(joinpath(@__DIR__, "G₂_gens.jl"))

G, roots, Weyl = G₂_roots_weyl()

const HALFRADIUS = 2
const UPPER_BOUND = Inf

RG, S, sizes = @time PropertyT.group_algebra(G, halfradius = HALFRADIUS)

Δ = RG(length(S)) - sum(RG(s) for s in S)

wd = let Σ = Weyl, RG = RG
    act = PropertyT.AlphabetPermutation{eltype(Σ),Int64}(
        Dict(g => PermutationGroups.perm(g) for g in Σ),
    )

    @time SymbolicWedderburn.WedderburnDecomposition(
        Float64,
        Σ,
        act,
        basis(RG),
        StarAlgebras.Basis{UInt16}(@view basis(RG)[1:sizes[HALFRADIUS]]),
        semisimple = false,
    )
end

elt = Δ^2
unit = Δ

@time model, varP = PropertyT.sos_problem_primal(
    elt,
    unit,
    wd;
    upper_bound = UPPER_BOUND,
    augmented = true,
    show_progress = true,
)
warm = nothing

begin
    @time status, warm = PropertyT.solve(
        model,
        scs_optimizer(;
            linear_solver = SCS.MKLDirectSolver,
            eps = 1e-10,
            max_iters = 20_000,
            accel = 50,
            alpha = 1.95,
        ),
        warm,
    )

    @info "reconstructing the solution"
    Q = @time begin
        wd = wd
        Ps = [JuMP.value.(P) for P in varP]
        if any(any(isnan, P) for P in Ps)
            throw("solver was probably interrupted, no valid solution available")
        end
        Qs = real.(sqrt.(Ps))
        PropertyT.reconstruct(Qs, wd)
    end
    P = Q' * Q

    @info "certifying the solution"
    @time certified, λ = PropertyT.certify_solution(
        elt,
        unit,
        JuMP.objective_value(model),
        Q;
        halfradius = HALFRADIUS,
        augmented = true,
    )
end

### grading below

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

Δs = PropertyT.laplacians(
    RG,
    S,
    x -> (gx = PropertyT.grading(x); Set([gx, -gx])),
)

elt = PropertyT.Adj(Δs)
elt == Δ^2 - PropertyT.Sq(Δs)
unit = Δ

@time model, varP = PropertyT.sos_problem_primal(
    elt,
    unit,
    wd;
    upper_bound = UPPER_BOUND,
    augmented = true,
)

warm = nothing

begin
    @time status, warm = PropertyT.solve(
        model,
        scs_optimizer(;
            linear_solver = SCS.MKLDirectSolver,
            eps = 1e-10,
            max_iters = 50_000,
            accel = 50,
            alpha = 1.95,
        ),
        warm,
    )

    @info "reconstructing the solution"
    Q = @time begin
        wd = wd
        Ps = [JuMP.value.(P) for P in varP]
        if any(any(isnan, P) for P in Ps)
            throw("solver was probably interrupted, no valid solution available")
        end
        Qs = real.(sqrt.(Ps))
        PropertyT.reconstruct(Qs, wd)
    end
    P = Q' * Q

    @info "certifying the solution"
    @time certified, λ = PropertyT.certify_solution(
        elt,
        unit,
        JuMP.objective_value(model),
        Q;
        halfradius = HALFRADIUS,
        augmented = true,
    )
end

# Δ² - 1 / 1 · Sq → -0.8818044647162608
# Δ² - 2 / 3 · Sq → -0.1031738
# Δ² - 1 / 2 · Sq → 0.228296213895906
# Δ² - 1 / 3 · Sq → 0.520
# Δ² - 0 / 1 · Sq → 0.9676851592000731
# Sq → 0.333423

# vals = [
#     1.0 -0.8818
#     2/3 -0.1032
#     1/2  0.2282
#     1/3  0.520
#     0    0.9677
# ]
