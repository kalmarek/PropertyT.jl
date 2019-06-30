__precompile__()
module PropertyT

using AbstractAlgebra
using LinearAlgebra
using SparseArrays
using Markdown
using Dates

using Groups
using GroupRings

using JLD
using JuMP

import AbstractAlgebra: Group, NCRing, perm

import MathProgBase.SolverInterface.AbstractMathProgSolver

include("laplacians.jl")
include("RGprojections.jl")
include("orbitdata.jl")
include("sos_sdps.jl")
include("checksolution.jl")
include("sqadjop.jl")

include("1712.07167.jl")


end # module Property(T)
