module Projections

using AbstractAlgebra
using Groups
using GroupRings
using Markdown

export PermCharacter, DirectProdCharacter, rankOne_projections

###############################################################################
#
#  Characters of Symmetric Group and DirectProduct
#
###############################################################################

abstract type AbstractCharacter end

struct PermCharacter <: AbstractCharacter
    p::Generic.Partition
end

struct DirectProdCharacter{N, T<:AbstractCharacter} <: AbstractCharacter
    chars::NTuple{N, T}
end

function (chi::DirectProdCharacter)(g::DirectPowerGroupElem)
    res = 1
    for (χ, elt) in zip(chi.chars, g.elts)
        res *= χ(elt)
    end
    return res
end

function (chi::PermCharacter)(g::Generic.perm)
    R = AbstractAlgebra.partitionseq(chi.p)
    p = Partition(Generic.permtype(g))
    return Int(Generic.MN1inner(R, p, 1, Generic._charvalsTable))
end

AbstractAlgebra.dim(χ::PermCharacter) = dim(YoungTableau(χ.p))

for T in [PermCharacter, DirectProdCharacter]
    @eval begin
        function (chi::$T)(X::GroupRingElem)
            RG = parent(X)
            z = zero(eltype(X))
            result = z
            for i in 1:length(X.coeffs)
                if X.coeffs[i] != z
                    result += chi(RG.basis[i])*X.coeffs[i]
                end
            end
            return result
        end
    end
end

characters(G::Generic.PermGroup) = (PermCharacter(p) for p in AllParts(G.n))

function characters(G::DirectPowerGroup{N}) where N
    nfold_chars = Iterators.repeated(characters(G.group), N)
    return (DirectProdCharacter(idx) for idx in Iterators.product(nfold_chars...))
end

###############################################################################
#
#  Projections
#
###############################################################################

function central_projection(RG::GroupRing, chi::AbstractCharacter, T::Type=Rational{Int})
    result = RG(zeros(T, length(RG.basis)))
    dim = chi(RG.group())
    ord = Int(order(RG.group))

    for g in RG.basis
        result[g] = convert(T, (dim//ord)*chi(g))
    end

    return result
end

function alternating_emb(RG::GroupRing{Gr,T}, V::Vector{T}, S::Type=Rational{Int}) where {Gr<:Generic.PermGroup, T<:GroupElem}
    res = RG(S)
    for g in V
        res[g] += sign(g)
    end
    return res
end

function idempotents(RG::GroupRing{Generic.PermGroup{S}}, T::Type=Rational{Int}) where S<:Integer
    if RG.group.n == 1
        return GroupRingElem{T}[one(RG,T)]
    elseif RG.group.n == 2
        Id = one(RG,T)
        transp = RG(perm"(1,2)", T)
        return GroupRingElem{T}[1//2*(Id + transp), 1//2*(Id - transp)]
    end

    projs = Vector{Vector{Generic.perm{S}}}()
    for l in 2:RG.group.n
        u = RG.group([circshift([i for i in 1:l], -1); [i for i in l+1:RG.group.n]])
        i = 0
        while (l-1)*i <= RG.group.n
            v = RG.group(circshift(collect(1:RG.group.n), i))
            k = inv(v)*u*v
            push!(projs, generateGroup([k], RG.group.n))
            i += 1
        end
    end

    idems = Vector{GroupRingElem{T}}()

    for p in projs
        append!(idems, [RG(p, T)//length(p), alternating_emb(RG, p, T)//length(p)])
    end

    return unique(idems)
end

function rankOne_projection(chi::PermCharacter, idems::Vector{T}) where {T<:GroupRingElem}

    RG = parent(first(idems))
    S = eltype(first(idems))

    ids = [one(RG, S); idems]
    zzz = zero(S)

    for (i,j,k) in Base.product(ids, ids, ids)
        if chi(i) == zzz || chi(j) == zzz || chi(k) == zzz
            continue
        else
            elt = i*j*k
            if elt^2 != elt
                continue
            elseif chi(elt) == one(S)
                return elt
                # return (i,j,k)
            end
        end
    end
    throw("Couldn't find rank-one projection for $chi")
end

function rankOne_projections(RG::GroupRing{G}, T::Type=Rational{Int}) where G<:Generic.PermGroup
    if RG.group.n == 1
        return [GroupRingElem([one(T)], RG)]
    end

    RGidems = idempotents(RG, T)

    min_projs = [central_projection(RG,chi)*rankOne_projection(chi,RGidems) for chi in characters(RG.group)]

    return min_projs
end

function ifelsetuple(a,b, k, n)
    x = [repeat([a], k); repeat([b], n-k)]
    return tuple(x...)
end

function orbit_selector(n::Integer, k::Integer,
        chi::AbstractCharacter, psi::AbstractCharacter)
    return Projections.DirectProdCharacter(ifelsetuple(chi, psi, k, n))
end

function rankOne_projections(RBn::GroupRing{G}, T::Type=Rational{Int}) where {G<:WreathProduct}

    Bn = RBn.group
    N = Bn.P.n
    # projections as elements of the group rings RSₙ
    Sn_rankOnePr = [rankOne_projections(
            GroupRing(PermGroup(i), collect(PermGroup(i))))
        for i in typeof(N)(1):N]

    # embedding into group ring of BN
    RN = GroupRing(Bn.N, collect(Bn.N))

    sign, id = collect(characters(Bn.N.group))
    # Bn.N = (Z/2Z)ⁿ characters corresponding to the first k coordinates:
    BnN_orbits = Dict(i => orbit_selector(N, i, sign, id) for i in 0:N)

    Q = Dict(i => RBn(g -> Bn(g), central_projection(RN, BnN_orbits[i], T)) for i in 0:N)
    Q = Dict(key => GroupRings.dense(val) for (key, val) in Q)

    all_projs = [Q[0]*RBn(g->Bn(g), p) for p in Sn_rankOnePr[N]]

    r = collect(1:N)
    for i in 1:N-1
        first_emb = g->Bn(Generic.emb!(Bn.P(), g, view(r, 1:i)))
        last_emb = g->Bn(Generic.emb!(Bn.P(), g, view(r, (i+1):N)))

        Sk_first = (RBn(first_emb, p) for p in Sn_rankOnePr[i])
        Sk_last = (RBn(last_emb, p) for p in Sn_rankOnePr[N-i])

        append!(all_projs,
        [Q[i]*p1*p2 for (p1,p2) in Base.product(Sk_first,Sk_last)])
    end

    append!(all_projs, [Q[N]*RBn(g->Bn(g), p) for p in Sn_rankOnePr[N]])

    return all_projs
end

##############################################################################
#
#   General Groups Misc
#
##############################################################################

@doc doc"""
    products(X::Vector{GroupElem}, Y::Vector{GroupElem}, op=*)
> Returns a vector of all possible products (or `op(x,y)`), where $x\in X$ and
> $y\in Y$ are group elements. You may specify which operation is used when
> forming 'products' by adding `op` (which is `*` by default).
"""
function products(X::AbstractVector{T}, Y::AbstractVector{T}, op=*) where {T<:GroupElem}
    result = Vector{T}()
    seen = Set{T}()
    for x in X
        for y in Y
            z = op(x,y)
            if !in(z, seen)
                push!(seen, z)
                push!(result, z)
            end
        end
    end
    return result
end

@doc doc"""
    generateGroup(gens::Vector{GroupElem}, r=2, Id=parent(first(gens))(), op=*)
> Produces all elements of a group generated by elements in `gens` in ball of
> radius `r` (word-length metric induced by `gens`).
> If `r(=2)` is specified the procedure will terminate after generating ball
> of radius `r` in the word-length metric induced by `gens`.
> The identity element `Id` and binary operation function `op` can be supplied
> to e.g. take advantage of additive group structure.
"""
function generateGroup(gens::Vector{T}, r=2, Id::T=parent(first(gens))(), op=*) where {T<:GroupElem}
    n = 0
    R = 1
    elts = gens
    gens = [Id; gens]
    while n ≠ length(elts) && R < r
        # @show elts
        R += 1
        n = length(elts)
        elts = products(elts, gens, op)
    end
    return elts
end

end # of module Projections