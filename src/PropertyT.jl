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

mutable struct Settings{Gr<:Group, GEl<:GroupElem, Sol<:AbstractMathProgSolver}
    name::String
    G::Gr
    S::Vector{GEl}
    radius::Int

    solver::Sol
    upper_bound::Float64
    tol::Float64
    warmstart::Bool

    autS::Group

    function Settings(name, G::Gr, S::Vector{GEl}, r::Int,
            sol::Sol, ub, tol, ws) where {Gr, GEl, Sol}
        return new{Gr, GEl, Sol}(name, G, S, r, sol, ub, tol, ws)
    end

    function Settings(name, G::Gr, S::Vector{GEl}, r::Int,
            sol::Sol, ub, tol, ws, autS) where {Gr, GEl, Sol}
        return new{Gr, GEl, Sol}(name, G, S, r, sol, ub, tol, ws, autS)
    end
end

prefix(s::Settings) = s.name
suffix(s::Settings) = "$(s.upper_bound)"
prepath(s::Settings) = prefix(s)
fullpath(s::Settings) = joinpath(prefix(s), suffix(s))

exists(fname::String) = isfile(fname) || islink(fname)

filename(prefix, s::Symbol) = filename(prefix, Val{s})

@eval begin
    for (s,n) in [
        [:fulllog,     "full_$(string(now())).log"],
        [:solverlog,   "solver_$(string(now())).log"],
        [:pm,          "pm.jld"],
        [:Δ,           "delta.jld"],
        [:λ,           "lambda.jld"],
        [:P,           "SDPmatrix.jld"],
        [:warm,        "warmstart.jld"],
        [:Uπs,         "U_pis.jld"],
        [:orbits,      "orbits.jld"],
        [:preps,       "preps.jld"],
    ]

        filename(prefix::String, ::Type{Val{$:(s)}}) = joinpath(prefix, :($n))
    end
end

for T in [:Naive, :Symmetrize]
    @eval begin
        function check_property_T(::Type{$T}, sett::Settings)

            if exists(filename(prepath(sett),:pm)) &&
                exists(filename(prepath(sett),:Δ))
                # cached
                Δ = loadLaplacian(prepath(sett), parent(sett.S[1]))
            else
                # compute
                Δ = computeLaplacian(sett.S, sett.radius)
                save(filename(prepath(sett), :pm), "pm", parent(Δ).pm)
                save(filename(prepath(sett), :Δ), "Δ", Δ.coeffs)
            end

            files_exist = exists(filename(fullpath(sett), :λ)) &&
                exists(filename(fullpath(sett), :P))

            if !sett.warmstart && files_exist
                λ, P = loadλandP(fullpath(sett))
            else
                warmfile = filename(fullpath(sett), :warm)
                if sett.warmstart && exists(warmfile)
                    ws = load(warmfile, "warmstart")
                else
                    ws = nothing
                end

                λ, P, ws = computeλandP($T, Δ, sett,
                    solverlog=filename(fullpath(sett), :solverlog))
                saveλandP(fullpath(sett), λ, P, ws)

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
    end
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


include("Laplacians.jl")
include("orbitdata.jl")
include("sos_sdps.jl")
include("CheckSolution.jl")


end # module Property(T)
