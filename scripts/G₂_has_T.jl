using LinearAlgebra
BLAS.set_num_threads(8)

ENV["OMP_NUM_THREADS"] = 4

using Groups
import Groups.MatrixGroups

include(joinpath(@__DIR__, "../test/optimizers.jl"))
using PropertyT

using PropertyT.SymbolicWedderburn
using PropertyT.PermutationGroups
using PropertyT.StarAlgebras

include(joinpath(@__DIR__, "argparse.jl"))
include(joinpath(@__DIR__, "utils.jl"))

# const N = parsed_args["N"]
const HALFRADIUS = parsed_args["halfradius"]
const UPPER_BOUND = parsed_args["upper_bound"]

include(joinpath(@__DIR__, "./G₂_gens.jl"))

G, roots, Weyl = G₂_roots_weyl()
@info "Running Δ² - λ·Δ sum of squares decomposition for G₂"

@info "computing group algebra structure"
RG, S, sizes = @time PropertyT.group_algebra(G, halfradius = HALFRADIUS)

@info "computing WedderburnDecomposition"
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
@info wd

Δ = RG(length(S)) - sum(RG(s) for s in S)
elt = Δ^2
unit = Δ

@time model, varP = PropertyT.sos_problem_primal(
    elt,
    unit,
    wd;
    upper_bound = UPPER_BOUND,
    augmented = true,
    show_progress = false,
)
warm = nothing
status = JuMP.OPTIMIZE_NOT_CALLED
certified, λ = false, nothing

while status ≠ JuMP.OPTIMAL
    @time status, warm = PropertyT.solve(
        model,
        scs_optimizer(;
            eps = 1e-10,
            max_iters = 20_000,
            accel = 50,
            alpha = 1.95,
        ),
        warm,
    )

    @info "reconstructing the solution"
    Q = @time let wd = wd, Ps = [JuMP.value.(P) for P in varP]
        Qs = real.(sqrt.(Ps))
        PropertyT.reconstruct(Qs, wd)
    end

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

if certified && λ > 0
    Κ(λ, S) = round(sqrt(2λ / length(S)), Base.RoundDown; digits = 5)
    @info "Certified result: G₂ has property (T):" N λ Κ(λ, S)
else
    @info "Could NOT certify the result:" certified λ
end
