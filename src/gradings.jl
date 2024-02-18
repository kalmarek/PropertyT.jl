## something about roots

function Roots.Root(e::MatrixGroups.ElementaryMatrix{N}) where {N}
    return Roots.𝕖(N, e.i) - Roots.𝕖(N, e.j)
end

function Roots.Root(s::MatrixGroups.ElementarySymplectic{N}) where {N}
    if s.symbol === :A
        return Roots.𝕖(N ÷ 2, s.i) - Roots.𝕖(N ÷ 2, s.j)
    else
        @assert s.symbol === :B
        n = N ÷ 2
        i, j = ifelse(s.i <= n, s.i, s.i - n), ifelse(s.j <= n, s.j, s.j - n)
        return (-1)^(s.i > s.j) * (Roots.𝕖(n, i) + Roots.𝕖(n, j))
    end
end

grading(s::MatrixGroups.ElementarySymplectic) = Roots.Root(s)
grading(e::MatrixGroups.ElementaryMatrix) = Roots.Root(e)

function grading(g::FPGroupElement)
    if length(word(g)) == 1
        A = alphabet(parent(g))
        return grading(A[first(word(g))])
    else
        throw("Grading is implemented only for generators")
    end
end

_groupby(f, iter::AbstractVector) = _groupby(f.(iter), iter)
function _groupby(keys::AbstractVector{K}, vals::AbstractVector{V}) where {K,V}
    @assert length(keys) == length(vals)
    d = Dict(k => V[] for k in keys)
    for (k, v) in zip(keys, vals)
        push!(d[k], v)
    end
    return d
end

function laplacians(RG::StarAlgebras.StarAlgebra, S, grading)
    d = _groupby(grading, S)
    Δs = Dict(α => RG(length(Sα)) - sum(RG(s) for s in Sα) for (α, Sα) in d)
    return Δs
end

function Adj(rootsystem::AbstractDict, subtype::Symbol)
    roots = let W = mapreduce(collect, union, keys(rootsystem))
        W = union!(W, -1 .* W)
    end

    return reduce(
        +,
        (
            Δα * Δβ for (α, Δα) in rootsystem for (β, Δβ) in rootsystem if
            Roots.classify_sub_root_system(roots, first(α), first(β)) == subtype
        );
        init = zero(first(values(rootsystem))),
    )
end

Adj(rootsystem::AbstractDict) = sum(values(rootsystem))^2 - Sq(rootsystem)

function Sq(rootsystem::AbstractDict)
    return reduce(
        +,
        Δα^2 for (_, Δα) in rootsystem;
        init = zero(first(values(rootsystem))),
    )
end
