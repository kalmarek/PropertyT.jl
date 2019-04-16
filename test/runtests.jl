using AbstractAlgebra, Nemo, Groups, SCS
using SparseArrays
using JLD
using PropertyT
using Test
using JuMP

indexing(n) = [(i,j) for i in 1:n for j in (i+1):n]
function Groups.gens(M::MatSpace)
    @assert ncols(M) == nrows(M)
    N = ncols(M)
    E(i,j) = begin g = M(1); g[i,j] = 1; g end
    S = [E(i,j) for (i,j) in indexing(N)]
    S = [S; transpose.(S)]
    return S
end

solver(iters; accel=1) =
    with_optimizer(SCS.Optimizer,
    linear_solver=SCS.Direct, max_iters=iters,
    acceleration_lookback=accel, eps=1e-10, warm_start=true)

include("1703.09680.jl")
include("1712.07167.jl")
include("SOS_correctness.jl")
