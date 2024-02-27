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

# fixes/hacks
import Groups.KnuthBendix
KnuthBendix.ordering(o::KnuthBendix.WordOrdering) = o
function KnuthBendix.rewrite!(
    u::KnuthBendix.AbstractWord,
    w::KnuthBendix.AbstractWord,
    o::KnuthBendix.WordOrdering,
)
    return KnuthBendix.rewrite!(u, w, KnuthBendix.alphabet(o))
end

struct Letter{T} <: Groups.GSymbol # letter of an Alphabet
    elt::T
end

Base.show(io::IO, tt::Letter) = show(io, tt.elt)
Base.inv(tt::Letter) = Letter(inv(tt.elt))
Base.:(==)(tt::Letter, ss::Letter) = tt.elt == ss.elt
Base.hash(tt::Letter, h::UInt) = hash(tt.elt, hash(Letter, h))

Base.Base.@propagate_inbounds function Groups.evaluate!(
    v::Tuple{Vararg{T,N}},
    tt::Letter,
    tmp = one(first(v)),
) where {T,N}
    return Groups.evaluate!(v, tt.elt, tmp)
end

function PropertyT._conj(tt::Letter, g)
    G = parent(tt.elt)
    A = alphabet(G)

    w = [A[PropertyT._conj(A[l], g)] for l in word(tt.elt)]
    return Letter(G(w))
end

G = let G = SpecialAutomorphismGroup(FreeGroup(N + 1))
    A = alphabet(G)
    lambdas = [Groups.λ(1, i) for i in 2:N+1]
    append!(lambdas, [Groups.λ(i, 1) for i in 2:N+1])
    rhos = [Groups.ϱ(1, i) for i in 2:N+1]
    append!(rhos, [Groups.ϱ(i, 1) for i in 2:N+1])

    _alph = eltype(G)[]

    for i in 2:N+1
        for j in 2:N+1
            i == j && continue
            g = G([A[Groups.ϱ(1, i)], A[Groups.ϱ(j, 1)]])
            h = G([A[Groups.λ(1, i)], A[Groups.λ(j, 1)]])
            push!(_alph, g, h)
        end
    end

    alph = Letter.(_alph)
    AutomorphismGroup(
        FreeGroup(N + 1),
        alph,
        KnuthBendix.LenLex(Groups.Alphabet(alph)),
        Groups.domain(one(G)),
    )
end
# @info "Running Δ² - λ·Δ sum of squares decomposition for " G

@info "computing group algebra structure"
RG, S, sizes = @time PropertyT.group_algebra(G, halfradius = HALFRADIUS)

@info "computing WedderburnDecomposition"
wd = let RG = RG, N = N
    G = StarAlgebras.object(RG)
    P = PermGroup(perm"(2,3)", Perm([1; 1 .+ circshift(1:N, -1)]))
    Σ = Groups.Constructions.WreathProduct(PermGroup(perm"(1,2)"), P)
    act = PropertyT.action_by_conjugation(G, P)

    wdfl = @time SW.WedderburnDecomposition(
        Float64,
        P,
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
