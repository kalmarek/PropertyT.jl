using Test
using LinearAlgebra, SparseArrays
using AbstractAlgebra, Groups, GroupRings
using PropertyT
using JLD

using JuMP, SCS

with_SCS(iters; accel=0, eps=1e-10, warm_start=true) =
    with_optimizer(SCS.Optimizer,
    linear_solver=SCS.DirectSolver, max_iters=iters,
    acceleration_lookback=accel, eps=eps, warm_start=warm_start)

include("1703.09680.jl")
include("actions.jl")
include("1712.07167.jl")
include("SOS_correctness.jl")
include("1812.03456.jl")
