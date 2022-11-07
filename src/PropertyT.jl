__precompile__()
module PropertyT

using LinearAlgebra
using SparseArrays
using Dates

using IntervalArithmetic
using JuMP

using Groups
using StarAlgebras
using SymbolicWedderburn

include("laplacians.jl")
include("constraint_matrix.jl")
include("sos_sdps.jl")
include("certify.jl")

include("sqadjop.jl")

include("1712.07167.jl")
include("1812.03456.jl")

end # module Property(T)
