## action induced from permuting letters of an alphabet

struct AlphabetPermutation{GEl,I} <: SymbolicWedderburn.ByPermutations
    perms::Dict{GEl,Perm{I}}
end

function AlphabetPermutation(
    A::Alphabet,
    Γ::PermutationGroups.AbstractPermutationGroup,
    op,
)
    return AlphabetPermutation(
        Dict(γ => inv(Perm([A[op(l, γ)] for l in A])) for γ in Γ),
    )
end

function AlphabetPermutation(A::Alphabet, W::Constructions.WreathProduct, op)
    return AlphabetPermutation(
        Dict(
            w => inv(Perm([A[op(op(l, w.p), w.n)] for l in A])) for
            w in W
        ),
    )
end

function SymbolicWedderburn.action(
    act::AlphabetPermutation,
    γ::GroupElement,
    w::Groups.AbstractWord,
)
    return w^(act.perms[γ])
end

function SymbolicWedderburn.action(
    act::AlphabetPermutation,
    γ::GroupElement,
    g::Groups.AbstractFPGroupElement,
)
    G = parent(g)
    w = word(g)^(act.perms[γ])
    return G(w)
end
