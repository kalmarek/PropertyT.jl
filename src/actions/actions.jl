import SymbolicWedderburn.action
StarAlgebras.star(g::Groups.GroupElement) = inv(g)

include("alphabet_permutation.jl")

include("sln_conjugation.jl")
include("spn_conjugation.jl")
include("autfn_conjugation.jl")

function SymbolicWedderburn.action(
    act::SymbolicWedderburn.ByPermutations,
    g::Groups.GroupElement,
    α::StarAlgebras.AlgebraElement
)
    res = StarAlgebras.zero!(similar(α))
    B = basis(parent(α))
    for (idx, val) in StarAlgebras._nzpairs(StarAlgebras.coeffs(α))
        a = B[idx]
        a_g = SymbolicWedderburn.action(act, g, a)
        res[a_g] += val
    end
    return res
end

function Base.:^(
    w::W,
    p::PermutationGroups.AbstractPerm,
) where {W<:Groups.AbstractWord}
    return W([l^p for l in w])
end
