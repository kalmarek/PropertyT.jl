__outer_dim(wd::WedderburnDecomposition) = size(first(direct_summands(wd)), 2)

function __group_of(wd::WedderburnDecomposition)
    # this is veeeery hacky... ;)
    return parent(first(keys(wd.hom.cache)))
end

function reconstruct(
    Ms::AbstractVector{<:AbstractMatrix},
    wbdec::WedderburnDecomposition;
    atol=eps(real(eltype(wbdec))) * 10__outer_dim(wbdec)
)
    n = __outer_dim(wbdec)
    res = sum(zip(Ms, SymbolicWedderburn.direct_summands(wbdec))) do (M, ds)
        res = similar(M, n, n)
        reconstruct!(res, M, ds, __group_of(wbdec), wbdec.hom, atol=atol)
    end
    return res
end

function reconstruct!(
    res::AbstractMatrix,
    M::AbstractMatrix,
    ds::SymbolicWedderburn.DirectSummand,
    G,
    hom;
    atol=eps(real(eltype(ds))) * 10max(size(res)...)
)
    res .= zero(eltype(res))
    U = SymbolicWedderburn.image_basis(ds)
    d = SymbolicWedderburn.degree(ds)
    tmp = (U' * M * U) .* d

    res = average!(res, tmp, G, hom)
    if eltype(res) <: AbstractFloat
        __droptol!(res, atol) # TODO: is this really necessary?!
    end
    return res
end

function __droptol!(M::AbstractMatrix, tol)
    for i in eachindex(M)
        if abs(M[i]) < tol
            M[i] = zero(M[i])
        end
    end
    return M
end

# implement average! for other actions when needed
function average!(
    res::AbstractMatrix,
    M::AbstractMatrix,
    G::Groups.Group,
    hom::SymbolicWedderburn.InducedActionHomomorphism{<:SymbolicWedderburn.ByPermutations}
)
    @assert size(M) == size(res)
    for g in G
        gext = SymbolicWedderburn.induce(hom, g)
        Threads.@threads for c in axes(res, 2)
            for r in axes(res, 1)
                res[r, c] += M[r^gext, c^gext]
            end
        end
    end
    o = Groups.order(Int, G)
    res ./= o
    return res
end
