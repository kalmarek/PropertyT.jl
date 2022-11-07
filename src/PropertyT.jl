__precompile__()
module PropertyT

using LinearAlgebra
using SparseArrays

using IntervalArithmetic
using JuMP

using Groups
import Groups.GroupsCore
using SymbolicWedderburn
import SymbolicWedderburn.StarAlgebras
import SymbolicWedderburn.PermutationGroups

include("constraint_matrix.jl")
include("sos_sdps.jl")
include("certify.jl")

include("sqadjop.jl")

include("roots.jl")
import .Roots
include("gradings.jl")

include("actions/actions.jl")

include("1712.07167.jl")
include("1812.03456.jl")

end # module Property(T)
