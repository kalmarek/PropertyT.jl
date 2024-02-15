using Test
using LinearAlgebra
using SparseArrays

using Groups
import Groups.MatrixGroups

using PropertyT
import SymbolicWedderburn as SW
import StarAlgebras as SA
import PermutationGroups as PG

include("optimizers.jl")
include("check_positivity.jl")
include("quick_tests.jl")

if haskey(ENV, "CI")
    @testset "PropertyT" begin
        include("constratint_matrices.jl")
        include("actions.jl")

        include("1703.09680.jl")
        include("1712.07167.jl")
        include("1812.03456.jl")

        include("roots.jl")
        include("graded_adj.jl")
        include("Chevalley.jl")
    end
end
