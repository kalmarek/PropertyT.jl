## Particular definitions for actions on Sp(n,ℤ)

function _conj(
    t::MatrixGroups.ElementarySymplectic{N,T},
    σ::PermutationGroups.AbstractPerm,
) where {N,T}
    @assert iseven(N)
    @assert degree(σ) == N ÷ 2 "Got degree = $(degree(σ)); N = $N"
    i = mod1(t.i, N ÷ 2)
    ib = i == t.i ? 0 : N ÷ 2
    j = mod1(t.j, N ÷ 2)
    jb = j == t.j ? 0 : N ÷ 2
    return MatrixGroups.ElementarySymplectic{N}(t.symbol, i^inv(σ) + ib, j^inv(σ) + jb, t.val)
end

function _conj(
    t::MatrixGroups.ElementarySymplectic{N,T},
    x::Groups.Constructions.DirectPowerElement,
) where {N,T}
    @assert Groups.order(Int, parent(x).group) == 2
    @assert iseven(N)
    just_one_flips = xor(isone(x.elts[mod1(t.i, N ÷ 2)]), isone(x.elts[mod1(t.j, N ÷ 2)]))
    return ifelse(just_one_flips, inv(t), t)
end

action_by_conjugation(sln::Groups.MatrixGroups.SymplecticGroup, Σ::Groups.Group) =
    AlphabetPermutation(alphabet(sln), Σ, _conj)
