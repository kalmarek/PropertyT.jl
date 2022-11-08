## Particular definitions for actions on Aut(F_n)

function _conj(
    t::Groups.Transvection,
    σ::PermutationGroups.AbstractPerm,
)
    return Groups.Transvection(t.id, t.i^inv(σ), t.j^inv(σ), t.inv)
end

function _flip(t::Groups.Transvection, g::Groups.GroupElement)
    isone(g) && return t
    return Groups.Transvection(t.id === :ϱ ? :λ : :ϱ, t.i, t.j, t.inv)
end

function _conj(
    t::Groups.Transvection,
    x::Groups.Constructions.DirectPowerElement,
)
    @assert Groups.order(Int, parent(x).group) == 2
    i, j = t.i, t.j
    t = ifelse(isone(x.elts[i] * x.elts[j]), t, inv(t))
    return _flip(t, x.elts[i])
end

action_by_conjugation(sautfn::Groups.AutomorphismGroup{<:Groups.FreeGroup}, Σ::Groups.Group) =
    AlphabetPermutation(alphabet(sautfn), Σ, _conj)
