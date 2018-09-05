include("Projections.jl")

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

function OrbitData(RG::GroupRing, autS::Group)
    orbs = orbit_decomposition(autS, RG.basis, RG.basis_dict)
    @assert sum(length(o) for o in orbs) == length(RG.basis)

    autS_mps = Projections.rankOne_projections(GroupRing(autS))

    preps = perm_reps(autS, RG.basis[1:size(RG.pm,1)], RG.basis_dict)
    mreps = matrix_reps(preps)

    Uπs = [orthSVD(matrix_repr(p, mreps)) for p in autS_mps]

    multiplicities = size.(Uπs,2)
    dimensions = [Int(p[autS()]*Int(order(autS))) for p in autS_mps]
    @assert dot(multiplicities, dimensions) == size(RG.pm,1)

    nzros = [i for i in 1:length(Uπs) if size(Uπs[i],2) !=0]

    return OrbitData(orbs, preps, Uπs[nzros], dims[nzros])
end

function compute_OrbitData(RG::GroupRing, autS::Group)

    info("Decomposing E into orbits of $(autS)")
    @time orbs = orbit_decomposition(autS, RG.basis, RG.basis_dict)
    @assert sum(length(o) for o in orbs) == length(RG.basis)
    info("E consists of $(length(orbs)) orbits!")

    info("Action matrices")
    @time preps = perm_reps(autS, RG.basis[1:size(RG.pm,1)], RG.basis_dict)
    mreps = matrix_reps(preps)

    info("Projections")
    @time autS_mps = Projections.rankOne_projections(GroupRing(autS));

    info("Uπs...")
    @time Uπs = [orthSVD(matrix_repr(p, mreps)) for p in autS_mps]

    multiplicities = size.(Uπs,2)
    info("multiplicities = $multiplicities")
    dimensions = [Int(p[autS()]*Int(order(autS))) for p in autS_mps];
    info("dimensions = $dimensions")
    @assert dot(multiplicities, dimensions) == size(RG.pm,1)

    return OrbitData(orbs, preps, Uπs, dimensions)
end

function decimate(od::OrbitData)
    nzros = [i for i in 1:length(od.Uπs) if size(od.Uπs[i],2) !=0]

    Us = map(x -> PropertyT.sparsify!(x, eps(Float64)*1e3, verbose=true), od.Uπs[nzros])
    #dimensions of the corresponding πs:
    dims = od.dims[nzros]

    return OrbitData(od.orbits, od.preps, full.(Us), dims);
end

function save_OrbitData(sett::Settings, data::OrbitData)
    save_preps(filename(prepath(sett), :preps), data.preps)

    save(filename(prepath(sett), :orbits),
        "orbits", data.orbits)

    save(filename(prepath(sett), :Uπs),
        "Uπs", data.Uπs,
        "dims", data.dims)
end

function load_OrbitData(sett::Settings)
    info("Loading Uπs, dims, orbits...")
    Uπs = load(filename(prepath(sett), :Uπs), "Uπs")
    nzros = [i for i in 1:length(Uπs) if size(Uπs[i],2) !=0]

    Uπs = map(x -> sparsify!(x, sett.tol/100, verbose=true), Uπs)
    #dimensions of the corresponding πs:
    dims = load(filename(prepath(sett), :Uπs), "dims")

    orbits = load(filename(prepath(sett), :orbits), "orbits")
    preps = load_preps(filename(prepath(sett), :preps), sett.autS)

    return OrbitData(orbits, preps, Uπs, dims)
end

function load_preps(fname::String, G::Group)
    lded_preps = load(fname, "perms_d")
    permG = PermutationGroup(length(first(lded_preps)))
    @assert length(lded_preps) == order(G)
    return Dict(k=>permG(v) for (k,v) in zip(elements(G), lded_preps))
end

function save_preps(fname::String, preps)
    autS = parent(first(keys(preps)))
    save(fname, "perms_d", [preps[elt].d for elt in elements(autS)])
end


###############################################################################
#
#  Sparsification
#
###############################################################################

dens(M::SparseMatrixCSC) = nnz(M)/length(M)
dens(M::AbstractArray) = countnz(M)/length(M)

function sparsify!{Tv,Ti}(M::SparseMatrixCSC{Tv,Ti}, eps=eps(Tv); verbose=false)

    densM = dens(M)
    for i in eachindex(M.nzval)
        if abs(M.nzval[i]) < eps
            M.nzval[i] = zero(Tv)
        end
    end
    dropzeros!(M)

    if verbose
        info("Sparsified density:", rpad(densM, 20), " → ", rpad(dens(M), 20), " ($(nnz(M)) non-zeros)")
    end

    return M
end

function sparsify!{T}(M::AbstractArray{T}, eps=eps(T); verbose=false)
    densM = dens(M)
    if verbose
        info("Sparsifying $(size(M))-matrix... ")
    end

    for n in eachindex(M)
        if abs(M[n]) < eps
            M[n] = zero(T)
        end
    end

    if verbose
        info("$(rpad(densM, 20)) → $(rpad(dens(M),20))), ($(countnz(M)) non-zeros)")
    end

    return sparse(M)
end

sparsify{T}(U::AbstractArray{T}, tol=eps(T); verbose=false) = sparsify!(deepcopy(U), tol, verbose=verbose)
