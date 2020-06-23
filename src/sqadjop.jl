isopposite(σ::Generic.Perm, τ::Generic.Perm, i=1, j=2) =
    σ[i] ≠ τ[i] && σ[i] ≠ τ[j] &&
    σ[j] ≠ τ[i] && σ[j] ≠ τ[j]

isadjacent(σ::Generic.Perm, τ::Generic.Perm, i=1, j=2) =
    (σ[i] == τ[i] && σ[j] ≠ τ[j]) || # first equal, second differ
    (σ[j] == τ[j] && σ[i] ≠ τ[i]) || # second equal, first differ
    (σ[i] == τ[j] && σ[j] ≠ τ[i]) || # first σ equal to second τ
    (σ[j] == τ[i] && σ[i] ≠ τ[j])    # second σ equal to first τ

Base.div(X::GroupRingElem, x::Number) = parent(X)(X.coeffs.÷x)

function Sq(RG::GroupRing, N::Integer)
    S₂ = generating_set(RG.group, 2)
    ℤ = Int64
    Δ₂ = length(S₂)*one(RG, ℤ) - RG(S₂, ℤ);

    Alt_N = [g for g in SymmetricGroup(N) if parity(g) == 0]

    sq = RG()
    for σ in Alt_N
        GroupRings.addeq!(sq, *(σ(Δ₂), σ(Δ₂), false))
    end
    return sq÷factorial(N-2)
end

function Adj(RG::GroupRing, N::Integer)
    S₂ = generating_set(RG.group, 2)
    ℤ = Int64
    Δ₂ = length(S₂)*one(RG, ℤ) - RG(S₂, ℤ);

    Alt_N = [g for g in SymmetricGroup(N) if parity(g) == 0]
    Δ₂s = Dict(σ=>σ(Δ₂) for σ in Alt_N)
    adj = RG()

    for σ in Alt_N
        for τ in Alt_N
            if isadjacent(σ, τ)
                GroupRings.addeq!(adj, *(Δ₂s[σ], Δ₂s[τ], false))
            end
        end
    end
    return adj÷factorial(N-2)^2
end

function Op(RG::GroupRing, N::Integer)
    if N < 4
        return RG()
    end
    S₂ = generating_set(RG.group, 2)
    ℤ = Int64
    Δ₂ = length(S₂)*one(RG, ℤ) - RG(S₂, ℤ);

    Alt_N = [g for g in SymmetricGroup(N) if parity(g) == 0]
    Δ₂s = Dict(σ=>σ(Δ₂) for σ in Alt_N)
    op = RG()

    for σ in Alt_N
        for τ in Alt_N
            if isopposite(σ, τ)
                GroupRings.addeq!(op, *(Δ₂s[σ], Δ₂s[τ], false))
            end
        end
    end
    return op÷factorial(N-2)^2
end

for Elt in [:Sq, :Adj, :Op]
    @eval begin
        $Elt(RG::GroupRing{AutGroup{N}}) where N = $Elt(RG, N)
        $Elt(RG::GroupRing{<:MatAlgebra}) = $Elt(RG, nrows(RG.group))
    end
end

function SqAdjOp(RG::GroupRing, N::Integer)
    S₂ = generating_set(RG.group, 2)
    ℤ = Int64
    Δ₂ = length(S₂)*one(RG, ℤ) - RG(S₂, ℤ);

    Alt_N = [σ for σ in SymmetricGroup(N) if parity(σ) == 0]
    sq, adj, op = RG(), RG(), RG()

    Δ₂s = Dict(σ=>σ(Δ₂) for σ in Alt_N)

    for σ in Alt_N
        GroupRings.addeq!(sq, *(Δ₂s[σ], Δ₂s[σ], false))
        for τ in Alt_N
            if isopposite(σ, τ)
                GroupRings.addeq!(op, *(Δ₂s[σ], Δ₂s[τ], false))
            elseif isadjacent(σ, τ)
                GroupRings.addeq!(adj, *(Δ₂s[σ], Δ₂s[τ], false))
            end
        end
    end

    k = factorial(N-2)
    return sq÷k, adj÷k^2, op÷k^2
end
