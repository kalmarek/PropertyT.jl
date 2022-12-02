using Test
using LinearAlgebra
using SparseArrays

using Groups
using Groups.GroupsCore
import Groups.MatrixGroups

using PropertyT
using SymbolicWedderburn
using SymbolicWedderburn.StarAlgebras
using SymbolicWedderburn.PermutationGroups

include("optimizers.jl")
include("check_positivity.jl")
include("quick_tests.jl")

if haskey(ENV, "FULL_TEST") || haskey(ENV, "CI")
    @testset "PropertyT" begin
        include("constratint_matrices.jl")
        include("actions.jl")

        include("1703.09680.jl")
        include("1712.07167.jl")
        include("1812.03456.jl")

        include("graded_adj.jl")
    end
end
