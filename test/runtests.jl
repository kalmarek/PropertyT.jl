using Test
using LinearAlgebra
using SparseArrays
BLAS.set_num_threads(1)
ENV["OMP_NUM_THREADS"] = 4

using Groups
using Groups.GroupsCore
import Groups.MatrixGroups

using PropertyT
using SymbolicWedderburn
using SymbolicWedderburn.StarAlgebras
using SymbolicWedderburn.PermutationGroups

include("optimizers.jl")

@testset "PropertyT" begin

    include("actions.jl")

    include("1703.09680.jl")
    include("1712.07167.jl")
    include("1812.03456.jl")

    include("graded_adj.jl")
end
