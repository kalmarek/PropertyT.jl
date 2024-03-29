# move to Groups
Base.keys(a::Alphabet) = keys(1:length(a))

## the old 1812.03456 definitions

function isopposite(
    σ::PG.AbstractPermutation,
    τ::PG.AbstractPermutation,
    i = 1,
    j = 2,
)
    return i^σ ≠ i^τ && i^σ ≠ j^τ && j^σ ≠ i^τ && j^σ ≠ j^τ
end

function isadjacent(
    σ::PG.AbstractPermutation,
    τ::PG.AbstractPermutation,
    i = 1,
    j = 2,
)
    return (i^σ == i^τ && j^σ ≠ j^τ) || # first equal, second differ
           (j^σ == j^τ && i^σ ≠ i^τ) || # second equal, first differ
           (i^σ == j^τ && j^σ ≠ i^τ) || # first σ equal to second τ
           (j^σ == i^τ && i^σ ≠ j^τ)    # second σ equal to first τ
end

function _ncycle(start, length, n=start + length - 1)
    p = Vector{UInt8}(1:n)
    @assert n ≥ start + length - 1
    for k in start:start+length-2
        p[k] = k + 1
    end
    p[start+length-1] = start
    return PG.Perm(p)
end

alternating_group(n::Integer) = PG.PermGroup([_ncycle(i, 3) for i in 1:n-2])

function small_gens(G::MatrixGroups.SpecialLinearGroup)
    A = alphabet(G)
    S = map([(1, 2), (2, 1)]) do (i, j)
        k = findfirst(l -> (l.i == i && l.j == j), A)
        @assert !isnothing(k)
        G(Groups.word_type(G)([k]))
    end
    return union!(S, inv.(S))
end

function small_gens(G::Groups.AutomorphismGroup{<:FreeGroup})
    A = alphabet(G)
    S = map([(:ϱ, 1, 2), (:ϱ, 2, 1), (:λ, 1, 2), (:λ, 2, 1)]) do (id, i, j)
        k = findfirst(l -> (l.id == id && l.i == i && l.j == j), A)
        @assert !isnothing(k)
        G(Groups.word_type(G)([k]))
    end
    return union!(S, inv.(S))
end

function small_laplacian(RG::SA.StarAlgebra)
    G = SA.object(RG)
    S₂ = small_gens(G)
    return length(S₂) * one(RG) - sum(RG(s) for s in S₂)
end

function SqAdjOp(A::SA.StarAlgebra, n::Integer, Δ₂ = small_laplacian(A))
    @assert parent(Δ₂) === A

    alt_n = alternating_group(n)
    G = SA.object(A)
    act = action_by_conjugation(G, alt_n)

    Δ₂s = Dict(σ => SW.action(act, σ, Δ₂) for σ in alt_n)

    sq, adj, op = zero(A), zero(A), zero(A)
    tmp = zero(A)

    for σ in alt_n
        SA.add!(sq, sq, SA.mul!(tmp, Δ₂s[σ], Δ₂s[σ]))
        for τ in alt_n
            if isopposite(σ, τ)
                SA.add!(op, op, SA.mul!(tmp, Δ₂s[σ], Δ₂s[τ]))
            elseif isadjacent(σ, τ)
                SA.add!(adj, adj, SA.mul!(tmp, Δ₂s[σ], Δ₂s[τ]))
            end
        end
    end

    k = factorial(n - 2)
    return sq ÷ k, adj ÷ k^2, op ÷ k^2
end
