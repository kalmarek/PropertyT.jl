module PropertyT

using LinearAlgebra
using SparseArrays

using JuMP

using Groups
import GroupsCore
using SymbolicWedderburn
import SymbolicWedderburn.StarAlgebras
import SymbolicWedderburn.PermutationGroups

include("constraint_matrix.jl")
include("sos_sdps.jl")
include("solve.jl")
include("reconstruct.jl")
include("certify.jl")

include("sqadjop.jl")

include("roots.jl")
import .Roots
include("gradings.jl")

include("actions/actions.jl")

function group_algebra(G::Groups.Group, S = gens(G); halfradius::Integer)
    S = union!(S, inv.(S))
    @info "generating wl-metric ball of radius $(2halfradius)"
    @time E, sizes = Groups.wlmetric_ball(S; radius = 2halfradius)
    @info "sizes = $(sizes)"
    @info "computing the *-algebra structure for G"
    @time RG = StarAlgebras.StarAlgebra(
        G,
        StarAlgebras.Basis{UInt32}(E),
        (sizes[halfradius], sizes[halfradius]);
        precompute = false,
    )
    return RG, S, sizes
end

end # module Property(T)
