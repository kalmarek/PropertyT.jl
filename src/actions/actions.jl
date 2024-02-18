include("alphabet_permutation.jl")

include("sln_conjugation.jl")
include("spn_conjugation.jl")
include("autfn_conjugation.jl")

function SW.action(
    act::SW.ByPermutations,
    g::Groups.GroupElement,
    α::SA.AlgebraElement,
)
    res = SA.zero!(similar(α))
    B = SA.basis(parent(α))
    for (idx, val) in SA._nzpairs(SA.coeffs(α))
        a = B[idx]
        a_g = SW.action(act, g, a)
        res[a_g] += val
    end
    return res
end

function Base.:^(w::W, p::PG.AbstractPermutation) where {W<:Groups.AbstractWord}
    return W([l^p for l in w])
end
