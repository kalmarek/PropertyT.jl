using JuMP
using SCS

export Settings, OrbitData


struct OrbitData{T<:AbstractArray{Float64, 2}}
    orbits::Vector{Vector{Int}}
    Uπs::Vector{T}
    dims::Vector{Int}
end

function OrbitData(sett::Settings)
    info("Loading Uπs, dims, orbits...")
    Uπs = load(filename(prepath(sett), :Uπs), "Uπs")
    nzros = [i for i in 1:length(Uπs) if size(Uπs[i],2) !=0]
    Uπs = Uπs[nzros]

    Uπs = map(x -> sparsify!(x, sett.tol/100, verbose=true), Uπs)
    #dimensions of the corresponding πs:
    dims = load(filename(prepath(sett), :Uπs), "dims")[nzros]

    orbits = load(filename(prepath(sett), :orbits), "orbits");

    return OrbitData(orbits, Uπs, dims)
end

include("OrbitDecomposition.jl")



end







end



end



end



    preps = load_preps(filename(prepath(sett), :preps), sett.autS)

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
