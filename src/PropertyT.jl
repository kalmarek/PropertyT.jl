__precompile__()
module PropertyT

using AbstractAlgebra
using Groups
using GroupRings

import AbstractAlgebra: Group, GroupElem, Ring, perm

using JLD
using JuMP

import MathProgBase.SolverInterface.AbstractMathProgSolver

include("laplacians.jl")
include("RGprojections.jl")
include("orbitdata.jl")
include("sos_sdps.jl")
include("checksolution.jl")

end # module Property(T)
