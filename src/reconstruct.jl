__outer_dim(wd::WedderburnDecomposition) = size(first(direct_summands(wd)), 2)

function __group_of(wd::WedderburnDecomposition)
    # this is veeeery hacky... ;)
    return parent(first(keys(wd.hom.cache)))
end

function reconstruct(
    Ms::AbstractVector{<:AbstractMatrix},
    wbdec::WedderburnDecomposition,
)
    n = __outer_dim(wbdec)
    res = sum(zip(Ms, SymbolicWedderburn.direct_summands(wbdec))) do (M, ds)
        res = similar(M, n, n)
        res = _reconstruct!(res, M, ds, __group_of(wbdec), wbdec.hom)
        return res
    end
    return res
end

function _reconstruct!(
    res::AbstractMatrix,
    M::AbstractMatrix,
    ds::SymbolicWedderburn.DirectSummand,
    G,
    hom,
)
    U = SymbolicWedderburn.image_basis(ds)
    d = SymbolicWedderburn.degree(ds)
    Θπ = (U' * M * U) .* d

    res .= zero(eltype(res))
    Θπ = average!(res, Θπ, G, hom)
    return Θπ
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
    hom::SymbolicWedderburn.InducedActionHomomorphism{
        <:SymbolicWedderburn.ByPermutations,
    },
)
    @assert size(M) == size(res)
    for g in G
        p = SymbolicWedderburn.induce(hom, g)
        Threads.@threads for c in axes(res, 2)
            # note: p is a permutation,
            # so accesses below are guaranteed to be disjoint
            for r in axes(res, 1)
                res[r^p, c^p] += M[r, c]
            end
        end
    end
    o = Groups.order(Int, G)
    res ./= o
    return res
end
