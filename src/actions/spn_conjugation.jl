## Particular definitions for actions on Sp(n,ℤ)

function _conj(
    s::MatrixGroups.ElementarySymplectic{N,T},
    σ::PG.AbstractPermutation,
) where {N,T}
    @assert iseven(N)
    @assert PG.degree(σ) ≤ N ÷ 2 "Got degree = $(PG.degree(σ)); N = $N"
    n = N ÷ 2
    @assert 1 ≤ s.i ≤ N
    @assert 1 ≤ s.j ≤ N
    if s.symbol == :A
        @assert 1 ≤ s.i ≤ n
        @assert 1 ≤ s.j ≤ n
        i = s.i^inv(σ)
        j = s.j^inv(σ)
    else
        @assert s.symbol == :B
        @assert xor(s.i > n, s.j > n)
        if s.i > n
            i = (s.i - n)^inv(σ) + n
            j = s.j^inv(σ)
        elseif s.j > n
            i = s.i^inv(σ)
            j = (s.j - n)^inv(σ) + n
        end
    end
    return MatrixGroups.ElementarySymplectic{N}(s.symbol, i, j, s.val)
end

function _conj(
    s::MatrixGroups.ElementarySymplectic{N,T},
    x::Groups.Constructions.DirectPowerElement,
) where {N,T}
    @assert Groups.order(Int, parent(x).group) == 2
    @assert iseven(N)
    n = N ÷ 2
    i, j = ifelse(s.i <= n, s.i, s.i - n), ifelse(s.j <= n, s.j, s.j - n)
    just_one_flips = xor(isone(x.elts[i]), isone(x.elts[j]))
    return ifelse(just_one_flips, inv(s), s)
end

function action_by_conjugation(
    sln::Groups.MatrixGroups.SymplecticGroup,
    Σ::Groups.Group,
)
    return AlphabetPermutation(alphabet(sln), Σ, _conj)
end
