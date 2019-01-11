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
    verbose && @info("Decomposing basis of RG into orbits of $(autS)")
    @time orbs = orbit_decomposition(autS, RG.basis, RG.basis_dict)
    @assert sum(length(o) for o in orbs) == length(RG.basis)
    verbose && @info("The action has $(length(orbs)) orbits")

    @time autS_mps = Projections.rankOne_projections(GroupRing(autS))
    verbose && @info("Projections in the Group Ring of AutS = $autS")

    verbose && @info("AutS-action matrix representatives")
    @time preps = perm_reps(autS, RG.basis[1:size(RG.pm,1)], RG.basis_dict)
    @time mreps = matrix_reps(preps)

    verbose && @info("Projection matrices Uπs")
    @time Uπs = [orthSVD(matrix_repr(p, mreps)) for p in autS_mps]

    multiplicities = size.(Uπs,2)
    verbose && @info("multiplicities = $multiplicities")
    dimensions = [Int(p[autS()]*Int(order(autS))) for p in autS_mps]
    verbose && @info("dimensions = $dimensions")
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

function orbit_decomposition(G::Group, E::Vector, rdict=GroupRings.reverse_dict(E))

    elts = collect(elements(G))

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
    if verbose
        @info("Sparsifying $(size(M))-matrix... ")
    end

    for n in eachindex(M)
        if abs(M[n]) < eps
            M[n] = zero(T)
        end
    end

    if verbose
        @info("$(rpad(densM, 20)) → $(rpad(dens(M),20))), ($(count(!iszero, M)) non-zeros)")
    end

    return sparse(M)
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
    elts = collect(elements(G))
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
