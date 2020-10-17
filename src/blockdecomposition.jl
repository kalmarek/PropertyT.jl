###############################################################################
#
#  BlockDecomposition
#
###############################################################################

struct BlockDecomposition{T<:AbstractArray{Float64, 2}, GEl<:GroupElem, P<:Generic.Perm}
    orbits::Vector{Vector{Int}}
    preps::Dict{GEl, P}
    Uπs::Vector{T}
    dims::Vector{Int}
end

function BlockDecomposition(RG::GroupRing, autS::Group, verbose=true)
    verbose && @info "Decomposing basis of RG into orbits of" autS
    @time orbs = orbit_decomposition(autS, RG.basis, RG.basis_dict)
    @assert sum(length(o) for o in orbs) == length(RG.basis)
    verbose && @info "The action has $(length(orbs)) orbits"

    verbose && @info "Finding projections in the Group Ring of" autS
    @time autS_mps = Projections.rankOne_projections(GroupRing(autS, collect(autS)))

    verbose && @info "Finding AutS-action matrix representation"
    @time preps = perm_reps(autS, RG.basis[1:size(RG.pm,1)], RG.basis_dict)
    @time mreps = matrix_reps(preps)

    verbose && @info "Computing the projection matrices Uπs"
    @time Uπs = [orthSVD(matrix_repr(p, mreps)) for p in autS_mps]

    multiplicities = size.(Uπs,2)
    dimensions = [Int(p[one(autS)]*Int(order(autS))) for p in autS_mps]
    if verbose
        info_strs = ["",
        lpad("multiplicities", 14) * "  =" * join(lpad.(multiplicities, 4), ""),
        lpad("dimensions", 14) * "  =" * join(lpad.(dimensions, 4), "")
        ]
        @info join(info_strs, "\n")
    end
    @assert dot(multiplicities, dimensions) == size(RG.pm,1)

    return BlockDecomposition(orbs, preps, Uπs, dimensions)
end

function decimate(od::BlockDecomposition, verbose=true)
    nzros = [i for i in 1:length(od.Uπs) if !isempty(od.Uπs[i])]

    Us = sparsify!.(od.Uπs, eps(Float64) * 1e4, verbose = verbose)[nzros]
    #dimensions of the corresponding Uπs:
    dims = od.dims[nzros]

    return BlockDecomposition(od.orbits, od.preps, Array{Float64}.(Us), dims)
end

function orthSVD(M::AbstractMatrix{T}) where {T<:AbstractFloat}
    fact = svd(convert(Matrix{T}, M))
    M_rank = sum(fact.S .> maximum(size(M)) * eps(T))
    return fact.U[:, 1:M_rank]
end

orbit_decomposition(
    G::Group,
    E::AbstractVector,
    rdict = GroupRings.reverse_dict(E);
    op = ^,
) = orbit_decomposition(collect(G), E, rdict; op=op)

function orbit_decomposition(elts::AbstractVector{<:GroupElem}, E::AbstractVector, rdict=GroupRings.reverse_dict(E); op=^)

    tovisit = trues(size(E));
    orbits = Vector{Vector{Int}}()

    orbit = zeros(Int, length(elts))

    for i in eachindex(E)
        if tovisit[i]
            g = E[i]
            Threads.@threads for j in eachindex(elts)
                orbit[j] = rdict[op(g, elts[j])]
            end
            tovisit[orbit] .= false
            push!(orbits, unique(orbit))
        end
    end
    return orbits
end

###############################################################################
#
#  Sparsification
#
###############################################################################

dens(M::SparseMatrixCSC) = nnz(M)/length(M)
dens(M::AbstractArray) = count(!iszero, M)/length(M)

function sparsify!(M::SparseMatrixCSC{Tv,Ti}, tol=eps(Tv); verbose=false) where {Tv,Ti}

    densM = dens(M)
    droptol!(M, tol)

    verbose && @info(
                "Sparsified density:",
                rpad(densM, 20),
                " → ",
                rpad(dens(M), 20),
                " ($(nnz(M)) non-zeros)"
            )

    return M
end

function sparsify!(M::AbstractArray{T}, tol=eps(T); verbose=false) where T
    densM = dens(M)
    clamp_small!(M, tol)

    if verbose
        @info("Sparsifying $(size(M))-matrix... \n $(rpad(densM, 20)) → $(rpad(dens(M),20))), ($(count(!iszero, M)) non-zeros)")
    end

    return sparse(M)
end

function clamp_small!(M::AbstractArray{T}, tol=eps(T)) where T
    for n in eachindex(M)
        if abs(M[n]) < tol
            M[n] = zero(T)
        end
    end
    return M
end

function sparsify(U::AbstractArray{T}, tol=eps(T); verbose=false) where T
    return sparsify!(deepcopy(U), tol, verbose=verbose)
end

###############################################################################
#
#  perm-, matrix-, representations
#
###############################################################################

function perm_repr(g::GroupElem, E::Vector, E_dict)
    p = Vector{Int}(undef, length(E))
    for (i,elt) in enumerate(E)
        p[i] = E_dict[elt^g]
    end
    return p
end

function perm_reps(G::Group, E::Vector, E_rdict=GroupRings.reverse_dict(E))
    elts = collect(G)
    l = length(elts)
    preps = Vector{Generic.Perm}(undef, l)

    permG = SymmetricGroup(length(E))

    Threads.@threads for i in 1:l
        preps[i] = permG(PropertyT.perm_repr(elts[i], E, E_rdict), false)
    end

    return Dict(elts[i]=>preps[i] for i in 1:l)
end

function matrix_repr(x::GroupRingElem, mreps::Dict)
    nzeros = findall(!iszero, x.coeffs)
    return sum(x[i].*mreps[parent(x).basis[i]] for i in nzeros)
end

function matrix_reps(preps::Dict{T,Generic.Perm{I}}) where {T<:GroupElem, I<:Integer}
    kk = collect(keys(preps))
    mreps = Vector{SparseMatrixCSC{Float64, Int}}(undef, length(kk))
    Threads.@threads for i in 1:length(kk)
        mreps[i] = AbstractAlgebra.matrix_repr(preps[kk[i]])
    end
    return Dict(kk[i] => mreps[i] for i in 1:length(kk))
end

###############################################################################
#
#  actions
#
###############################################################################

function Base.:^(y::GroupRingElem, g::GroupRingElem, op = ^)
    res = parent(y)()
    for elt in GroupRings.supp(g)
        res += g[elt] * ^(y, elt, op)
    end
    return res
end

function Base.:^(y::GroupRingElem, g::GroupElem, op = ^)
    RG = parent(y)
    result = zero(RG, eltype(y.coeffs))

    for (idx, c) in enumerate(y.coeffs)
        if !iszero(c)
            result[op(RG.basis[idx], g)] = c
        end
    end
    return result
end

function Base.:^(
    y::GroupRingElem{T,<:SparseVector},
    g::GroupElem,
    op = ^,
) where {T}
    RG = parent(y)
    index = [RG.basis_dict[op(RG.basis[idx], g)] for idx in y.coeffs.nzind]

    result = GroupRingElem(sparsevec(index, y.coeffs.nzval, y.coeffs.n), RG)

    return result
end

###############################################################################
#
#  perm && WreathProductElems actions: MatAlgElem
#
###############################################################################

function Base.:^(A::MatAlgElem, p::Generic.Perm)
    length(p.d) == size(A, 1) == size(A, 2) ||
        throw("Can't act via $p on matrix of size $(size(A))")
    result = similar(A)
    @inbounds for i = 1:size(A, 1)
        for j = 1:size(A, 2)
            result[p[i], p[j]] = A[i, j] # action by permuting rows and colums/conjugation
        end
    end
    return result
end

function Base.:^(A::MatAlgElem, g::WreathProductElem{N}) where {N}
    # @assert N == size(A,1) == size(A,2)
    flips = ntuple(i -> (g.n[i].d[1] == 1 && g.n[i].d[2] == 2 ? 1 : -1), N)
    result = similar(A)
    R = base_ring(parent(A))
    tmp = R(1)

    @inbounds for i = 1:size(A, 1)
        for j = 1:size(A, 2)
            x = A[i, j]
            if flips[i] * flips[j] == 1
                result[g.p[i], g.p[j]] = x
            else
                result[g.p[i], g.p[j]] = -x
            end
        end
    end
    return result
end

###############################################################################
#
# perm && WreathProductElems actions: Automorphism
#
###############################################################################

function Base.:^(a::Automorphism, g::GroupElem)
    Ag = parent(a)(g)
    Ag_inv = inv(Ag)
    res = append!(Ag, a, Ag_inv)
    return Groups.freereduce!(res)
end

(A::AutGroup)(p::Generic.Perm) = A(Groups.AutSymbol(p))

function (A::AutGroup)(g::WreathProductElem)
    isa(A.objectGroup, FreeGroup) || throw("Not an Aut(Fₙ)")
    parent(g).P.n == length(A.objectGroup.gens) ||
        throw("No natural embedding of $(parent(g)) into $A")
    elt = one(A)
    Id = one(parent(g.n.elts[1]))
    for i = 1:length(g.p.d)
        if g.n.elts[i] != Id
            push!(elt, Groups.flip(i))
        end
    end
    push!(elt, Groups.AutSymbol(g.p))
    return elt
end


# fallback:
Base.one(p::Generic.Perm) = Perm(length(p.d))
