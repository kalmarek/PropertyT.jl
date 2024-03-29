function __outer_dim(wd::SW.WedderburnDecomposition)
    return size(first(SW.direct_summands(wd)), 2)
end

function __group_of(wd::SW.WedderburnDecomposition)
    # this is veeeery hacky... ;)
    return parent(first(keys(wd.hom.cache)))
end

function reconstruct(
    Ms::AbstractVector{<:AbstractMatrix},
    wbdec::SW.WedderburnDecomposition,
)
    n = __outer_dim(wbdec)
    res = zeros(eltype(first(Ms)), n, n)
    for (M, ds) in zip(Ms, SW.direct_summands(wbdec))
        res = _reconstruct!(res, M, ds)
    end
    res = average!(zero(res), res, __group_of(wbdec), wbdec.hom)
    return res
end

function _reconstruct!(
    res::AbstractMatrix,
    M::AbstractMatrix,
    ds::SW.DirectSummand,
)
    if !iszero(M)
        U = SW.image_basis(ds)
        d = SW.degree(ds)
        res .+= (U' * M * U) .* d
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
    hom::SW.InducedActionHomomorphism{
        <:SW.ByPermutations,
    },
)
    res .= zero(eltype(res))
    @assert size(M) == size(res)
    o = Groups.order(Int, G)
    for g in G
        p = SW.induce(hom, g)
        Threads.@threads for c in axes(res, 2)
            for r in axes(res, 1)
                if !iszero(M[r, c])
                    res[r^p, c^p] += M[r, c] / o
                end
            end
        end
    end
    return res
end
