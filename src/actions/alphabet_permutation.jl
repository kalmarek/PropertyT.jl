## action induced from permuting letters of an alphabet

import Groups: Constructions

struct AlphabetPermutation{GEl,I} <: SymbolicWedderburn.ByPermutations
    perms::Dict{GEl,PermutationGroups.Perm{I}}
end

function AlphabetPermutation(
    A::Alphabet,
    Γ::PermutationGroups.AbstractPermutationGroup,
    op,
)
    return AlphabetPermutation(
        Dict(γ => inv(PermutationGroups.Perm([A[op(l, γ)] for l in A])) for γ in Γ),
    )
end

function AlphabetPermutation(A::Alphabet, W::Constructions.WreathProduct, op)
    return AlphabetPermutation(
        Dict(
            w => inv(PermutationGroups.Perm([A[op(op(l, w.p), w.n)] for l in A])) for
            w in W
        ),
    )
end

function SymbolicWedderburn.action(
    act::AlphabetPermutation,
    γ::Groups.GroupElement,
    g::Groups.AbstractFPGroupElement,
)
    G = parent(g)
    w = SymbolicWedderburn.action(act, γ, word(g))
    return G(w)
end

function SymbolicWedderburn.action(
    act::AlphabetPermutation,
    γ::Groups.GroupElement,
    w::Groups.AbstractWord,
)
    return w^(act.perms[γ])
end
