## action induced from permuting letters of an alphabet

import Groups: Constructions

struct AlphabetPermutation{GEl,I} <: SW.ByPermutations
    perms::Dict{GEl,PG.Perm{I}}
end

function AlphabetPermutation(
    A::Alphabet,
    Γ::PG.AbstractPermutationGroup,
    op,
)
    return AlphabetPermutation(
        Dict(γ => inv(PG.Perm([A[op(l, γ)] for l in A])) for γ in Γ),
    )
end

function AlphabetPermutation(A::Alphabet, W::Constructions.WreathProduct, op)
    return AlphabetPermutation(
        Dict(
            w => inv(PG.Perm([A[op(op(l, w.p), w.n)] for l in A])) for
            w in W
        ),
    )
end

function SW.action(
    act::AlphabetPermutation,
    γ::Groups.GroupElement,
    g::Groups.AbstractFPGroupElement,
)
    G = parent(g)
    # w = SW.action(act, γ, word(g))
    w = word(g)^(act.perms[γ])
    return G(w)
end

