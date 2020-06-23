indexing(n) = [(i,j) for i in 1:n for j in 1:n if i≠j]

function generating_set(G::AutGroup{N}, n=N) where N

    rmuls = [Groups.transvection_R(i,j) for (i,j) in indexing(n)]
    lmuls = [Groups.transvection_L(i,j) for (i,j) in indexing(n)]
    gen_set = G.([rmuls; lmuls])

    return [gen_set; inv.(gen_set)]
end

function EltaryMat(M::MatAlgebra, i::Integer, j::Integer, val=1)
    @assert i ≠ j
    @assert 1 ≤ i ≤ nrows(M)
    @assert 1 ≤ j ≤ ncols(M)
    m = one(M)
    m[i,j] = val
    return m
end

function generating_set(M::MatAlgebra, n=nrows(M))
    elts = [EltaryMat(M, i,j) for (i,j) in indexing(n)]
    return elem_type(M)[elts; inv.(elts)]
end

include("sqadjop.jl")
