__precompile__()
module PropertyT

using LinearAlgebra
using SparseArrays
using Dates

using JuMP

using Groups
using StarAlgebras
using SymbolicWedderburn

include("laplacians.jl")
include("sos_sdps.jl")
include("checksolution.jl")

include("1712.07167.jl")
include("1812.03456.jl")

end # module Property(T)
