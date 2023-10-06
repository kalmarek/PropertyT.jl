## Particular definitions for actions on SL(n,ℤ)

function _conj(
    t::MatrixGroups.ElementaryMatrix{N},
    σ::PermutationGroups.AbstractPermutation,
) where {N}
    return MatrixGroups.ElementaryMatrix{N}(t.i^inv(σ), t.j^inv(σ), t.val)
end

function _conj(
    t::MatrixGroups.ElementaryMatrix{N},
    x::Groups.Constructions.DirectPowerElement,
) where {N}
    @assert Groups.order(Int, parent(x).group) == 2
    just_one_flips = xor(isone(x.elts[t.i]), isone(x.elts[t.j]))
    return ifelse(just_one_flips, inv(t), t)
end

action_by_conjugation(sln::Groups.MatrixGroups.SpecialLinearGroup, Σ::Groups.Group) =
    AlphabetPermutation(alphabet(sln), Σ, _conj)
