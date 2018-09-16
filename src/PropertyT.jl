__precompile__()
module PropertyT

using AbstractAlgebra
using Groups
using GroupRings

import AbstractAlgebra: Group, GroupElem, Ring, perm

using JLD
using JuMP

import MathProgBase.SolverInterface.AbstractMathProgSolver

###############################################################################
#
#  Settings and filenames
#
###############################################################################

struct Symmetrize end
struct Naive end

struct Settings{T, GEl<:GroupElem}
    name::String

    G::Group
    S::Vector{GEl}
    radius::Int

    solver::AbstractMathProgSolver
    upper_bound::Float64
    tol::Float64
    warmstart::Bool

    autS::Group

    function Settings(name::String,
        G::Group, S::Vector{GEl}, r::Int,
        sol::Sol, ub, tol, ws) where {GEl<:GroupElem, Sol<:AbstractMathProgSolver}
        return new{Naive, GEl}(name, G, S, r, sol, ub, tol, ws)
    end

    function Settings(name::String,
        G::Group, S::Vector{GEl}, r::Int,
        sol::Sol, ub, tol, ws, autS) where {GEl<:GroupElem, Sol<:AbstractMathProgSolver}
        return new{Symmetrize, GEl}(name, G, S, r, sol, ub, tol, ws, autS)
    end
end


prefix(s::Settings{Naive}) = s.name
prefix(s::Settings{Symmetrize}) = "o"*s.name
suffix(s::Settings) = "$(s.upper_bound)"
prepath(s::Settings) = prefix(s)
fullpath(s::Settings) = joinpath(prefix(s), suffix(s))

filename(sett::Settings, s::Symbol) = filename(sett, Val{s})

filename(sett::Settings, ::Type{Val{:fulllog}}) =
    joinpath(fullpath(sett), "full_$(string(now())).log")
filename(sett::Settings, ::Type{Val{:solverlog}}) =
    joinpath(fullpath(sett), "solver_$(string(now())).log")

filename(sett::Settings, ::Type{Val{:Δ}}) =
    joinpath(prepath(sett), "delta.jld")
filename(sett::Settings, ::Type{Val{:OrbitData}}) =
    joinpath(prepath(sett), "OrbitData.jld")

filename(sett::Settings, ::Type{Val{:warmstart}}) =
    joinpath(fullpath(sett), "warmstart.jld")
filename(sett::Settings, ::Type{Val{:solution}}) =
    joinpath(fullpath(sett), "solution.jld")

function check_property_T(sett::Settings)
    fp = PropertyT.fullpath(sett)
    isdir(fp) || mkpath(fp)

    if isfile(filename(sett,:Δ))
        # cached
        Δ = loadLaplacian(filename(sett,:Δ), sett.G)
    else
        # compute
        Δ = Laplacian(sett.S, sett.radius)
        saveLaplacian(filename(sett, :Δ), Δ)
    end

    if !sett.warmstart && isfile(filename(sett, :solution))
        λ, P = load(filename(sett, :solution), "λ", "P")
    else
        λ, P = computeλandP(sett, Δ,
            solverlog=filename(sett, :solverlog))

        save(filename(sett, :solution), "λ", λ, "P", P)

        if λ < 0
            warn("Solver did not produce a valid solution!")
        end
    end

    info("λ = $λ")
    info("sum(P) = $(sum(P))")
    info("maximum(P) = $(maximum(P))")
    info("minimum(P) = $(minimum(P))")

    isapprox(eigvals(P), abs.(eigvals(P)), atol=sett.tol) ||
        warn("The solution matrix doesn't seem to be positive definite!")

    @time Q = real(sqrtm((P+P')/2))
    sgap = distance_to_cone(Δ, λ, Q, wlen=2*sett.radius)

    return interpret_results(sett, sgap)
end

Kazhdan(λ::Number, N::Integer) = sqrt(2*λ/N)

function interpret_results(sett::Settings, sgap::Number)

    if sgap > 0
        Kazhdan_κ = Kazhdan(sgap, length(sett.S))
        if Kazhdan_κ > 0
            info("κ($(sett.name), S) ≥ $Kazhdan_κ: Group HAS property (T)!")
            return true
        end
    end
    info("λ($(sett.name), S) ≥ $sgap < 0: Tells us nothing about property (T)")
    return false
end

include("laplacians.jl")
include("RGprojections.jl")
include("orbitdata.jl")
include("sos_sdps.jl")
include("checksolution.jl")


end # module Property(T)
