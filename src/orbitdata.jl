###############################################################################
#
#  OrbitData
#
###############################################################################

struct OrbitData{T<:AbstractArray{Float64, 2}, GEl<:GroupElem, P<:perm}
    orbits::Vector{Vector{Int}}
    preps::Dict{GEl, P}
    Uπs::Vector{T}
    dims::Vector{Int}
end

function OrbitData(RG::GroupRing, autS::Group, verbose=true)
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
    dimensions = [Int(p[autS()]*Int(order(autS))) for p in autS_mps]
    if verbose
        info_strs = ["",
        lpad("multiplicities", 14) * "  =" * join(lpad.(multiplicities, 4), ""),
        lpad("dimensions", 14) * "  =" * join(lpad.(dimensions, 4), "")
        ]
        @info join(info_strs, "\n")
    end
    @assert dot(multiplicities, dimensions) == size(RG.pm,1)

    return OrbitData(orbs, preps, Uπs, dimensions)
end

function decimate(od::OrbitData)
    nzros = [i for i in 1:length(od.Uπs) if size(od.Uπs[i],2) !=0]

    Us = map(x -> PropertyT.sparsify!(x, eps(Float64)*1e3, verbose=true), od.Uπs[nzros])
    #dimensions of the corresponding πs:
    dims = od.dims[nzros]

    return OrbitData(od.orbits, od.preps, Array{Float64}.(Us), dims);
end

function orthSVD(M::AbstractMatrix{T}) where {T<:AbstractFloat}
    M = Matrix(M)
    fact = svd(M)
    M_rank = sum(fact.S .> maximum(size(M))*eps(T))
    return fact.U[:,1:M_rank]
end

orbit_decomposition(G::Group, E::AbstractVector, rdict=GroupRings.reverse_dict(E)) = orbit_decomposition(collect(G), E, rdict)

function orbit_decomposition(elts::AbstractVector{<:GroupElem}, E::AbstractVector, rdict=GroupRings.reverse_dict(E))

    tovisit = trues(size(E));
    orbits = Vector{Vector{Int}}()

    orbit = zeros(Int, length(elts))

    for i in eachindex(E)
        if tovisit[i]
            g = E[i]
            Threads.@threads for j in eachindex(elts)
                orbit[j] = rdict[elts[j](g)]
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

function sparsify!(M::SparseMatrixCSC{Tv,Ti}, eps=eps(Tv); verbose=false) where {Tv,Ti}

    densM = dens(M)
    for i in eachindex(M.nzval)
        if abs(M.nzval[i]) < eps
            M.nzval[i] = zero(Tv)
        end
    end
    dropzeros!(M)

    if verbose
        @info("Sparsified density:", rpad(densM, 20), " → ", rpad(dens(M), 20), " ($(nnz(M)) non-zeros)")
    end

    return M
end

function sparsify!(M::AbstractArray{T}, eps=eps(T); verbose=false) where T
    densM = dens(M)
    clamp_small!(M, eps)

    if verbose
        @info("Sparsifying $(size(M))-matrix... \n $(rpad(densM, 20)) → $(rpad(dens(M),20))), ($(count(!iszero, M)) non-zeros)")
    end

    return sparse(M)
end

function clamp_small!(M::AbstractArray{T}, eps=eps(T)) where T
    for n in eachindex(M)
        if abs(M[n]) < eps
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
        p[i] = E_dict[g(elt)]
    end
    return p
end

function perm_reps(G::Group, E::Vector, E_rdict=GroupRings.reverse_dict(E))
    elts = collect(G)
    l = length(elts)
    preps = Vector{perm}(undef, l)

    permG = PermutationGroup(length(E))

    Threads.@threads for i in 1:l
        preps[i] = permG(PropertyT.perm_repr(elts[i], E, E_rdict), false)
    end

    return Dict(elts[i]=>preps[i] for i in 1:l)
end

function matrix_repr(x::GroupRingElem, mreps::Dict)
    nzeros = findall(!iszero, x.coeffs)
    return sum(x[i].*mreps[parent(x).basis[i]] for i in nzeros)
end

function matrix_reps(preps::Dict{T,perm{I}}) where {T<:GroupElem, I<:Integer}
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

function (g::GroupRingElem)(y::GroupRingElem)
    res = parent(y)()
    for elt in GroupRings.supp(g)
        res += g[elt]*elt(y)
    end
    return res
end

###############################################################################
#
#  perm actions
#
###############################################################################

function (g::perm)(y::GroupRingElem)
    RG = parent(y)
    result = zero(RG, eltype(y.coeffs))

    for (idx, c) in enumerate(y.coeffs)
        if c!= zero(eltype(y.coeffs))
            result[g(RG.basis[idx])] = c
        end
    end
    return result
end

function (g::perm)(y::GroupRingElem{T, <:SparseVector}) where T
    RG = parent(y)
    index = [RG.basis_dict[g(RG.basis[idx])] for idx in y.coeffs.nzind]

    result = GroupRingElem(sparsevec(index, y.coeffs.nzval, y.coeffs.n), RG)

    return result
end

function (p::perm)(A::MatElem)
    length(p.d) == size(A, 1) == size(A,2) || throw("Can't act via $p on matrix of size $(size(A))")
    result = similar(A)
    @inbounds for i in 1:size(A, 1)
        for j in 1:size(A, 2)
            result[p[i],p[j]] = A[i,j] # action by permuting rows and colums/conjugation
        end
    end
    return result
end

function (p::perm)(A::MatElem)
    length(p.d) == size(A, 1) == size(A,2) || throw("Can't act via $p on matrix of size $(size(A))")
    result = similar(A)
    @inbounds for i in 1:size(A, 1)
        for j in 1:size(A, 2)
            result[i, j] = A[p[i], p[j]] # action by permuting rows and colums/conjugation
        end
    return result
end

###############################################################################
#
#  WreathProductElems action on Nemo.MatElem
#
###############################################################################

function (g::WreathProductElem)(y::GroupRingElem)
    RG = parent(y)
    result = zero(RG, eltype(y.coeffs))

    for (idx, c) in enumerate(y.coeffs)
        if c!= zero(eltype(y.coeffs))
            result[g(RG.basis[idx])] = c
        end
    end
    return result
end

function (g::WreathProductElem{N})(A::MatElem) where N
    # @assert N == size(A,1) == size(A,2)
    flips = ntuple(i->(g.n[i].d[1]==1 && g.n[i].d[2]==2 ? 1 : -1), N)
    result = similar(A)
    @inbounds for i = 1:size(A,1)
        for j = 1:size(A,2)
            result[i, j] = A[g.p[i], g.p[j]]*(flips[i]*flips[j])
        end
    end
    return result
end

###############################################################################
#
#  Action of WreathProductElems on AutGroupElem
#
###############################################################################

function AutFG_emb(A::AutGroup, g::WreathProductElem)
    isa(A.objectGroup, FreeGroup) || throw("Not an Aut(Fₙ)")
    parent(g).P.n == length(A.objectGroup.gens) || throw("No natural embedding of $(parent(g)) into $A")
    elt = A()
    Id = parent(g.n.elts[1])()
    flips = Groups.AutSymbol[Groups.flip_autsymbol(i) for i in 1:length(g.p.d) if g.n.elts[i] != Id]
    Groups.r_multiply!(elt, flips, reduced=false)
    Groups.r_multiply!(elt, [Groups.perm_autsymbol(g.p)])
    return elt
end

function AutFG_emb(A::AutGroup, p::perm)
    isa(A.objectGroup, FreeGroup) || throw("Not an Aut(Fₙ)")
    parent(p).n == length(A.objectGroup.gens) || throw("No natural embedding of $(parent(p)) into $A")
    return A(Groups.perm_autsymbol(p))
end

function (g::WreathProductElem)(a::Groups.Automorphism)
    A = parent(a)
    g = AutFG_emb(A,g)
    res = A()
    Groups.r_multiply!(res, g.symbols, reduced=false)
    Groups.r_multiply!(res, a.symbols, reduced=false)
    Groups.r_multiply!(res, [inv(s) for s in reverse!(g.symbols)])
    return res
end

function (p::perm)(a::Groups.Automorphism)
    g = AutFG_emb(parent(a),p)
    return g*a*inv(g)
end
