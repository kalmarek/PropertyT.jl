__precompile__()
module PropertyT

using AbstractAlgebra
using Groups
using GroupRings

import AbstractAlgebra: Group, GroupElem, Ring, perm

using JLD
using JuMP
using MathProgBase

exists(fname::String) = isfile(fname) || islink(fname)

filename(prefix, s::Symbol) = filename(prefix, Val{s})

@eval begin
    for (s,n) in [
        [:pm,   "pm.jld"],
        [:Δ,    "delta.jld"],
        [:λ,    "lambda.jld"],
        [:P,    "SDPmatrix.jld"],
        [:warm, "warmstart.jld"],
        [:Uπs,  "U_pis.jld"],
        [:orb,  "orbits.jld"],
        [:preps,"preps.jld"],

        [:fulllog,   "full_$(string(now())).log"],
        [:solverlog,   "solver_$(string(now())).log"]
        ]

        filename(prefix::String, ::Type{Val{$:(s)}}) = joinpath(prefix, :($n))
    end
end

function loadLaplacian(name::String, G::Group)
    if exists(filename(name, :Δ)) && exists(filename(name, :pm))
        info("Loading precomputed Δ...")
        RG = GroupRing(G, load(filename(name, :pm), "pm"))
        Δ = GroupRingElem(load(filename(name, :Δ), "Δ")[:, 1], RG)
    else
        throw("You need to precompute $(filename(name, :pm)) and $(filename(name, :Δ)) to load it!")
    end
    return Δ
end

function computeLaplacian(S::Vector{E}, radius) where E<:AbstractAlgebra.RingElem
    R = parent(first(S))
    return computeLaplacian(S, one(R), radius)
end

function computeLaplacian(S::Vector{E}, radius) where E<:AbstractAlgebra.GroupElem
    G = parent(first(S))
    return computeLaplacian(S, G(), radius)
end

function computeLaplacian(S, Id, radius)
    info("Generating metric ball of radius $radius...")
    @time E_R, sizes = Groups.generate_balls(S, Id, radius=2radius)
    info("Generated balls of sizes $sizes.")

    info("Creating product matrix...")
    @time pm = GroupRings.create_pm(E_R, GroupRings.reverse_dict(E_R), sizes[radius]; twisted=true)

    RG = GroupRing(parent(Id), E_R, pm)
    Δ = spLaplacian(RG, S)
    return Δ
end

function loadλandP(name::String)
    λ_fname = filename(name, :λ)
    P_fname = filename(name, :P)

    if exists(λ_fname) && exists(P_fname)
        info("Loading precomputed λ, P...")
        λ = load(λ_fname, "λ")
        P = load(P_fname, "P")
    else
        throw("You need to precompute $λ_fname and $P_fname to load it!")
    end
    return λ, P
end

function computeλandP(Δ::GroupRingElem, upper_bound::AbstractFloat, solver, ws=nothing; solverlog=tempname()*".log")
    info("Creating SDP problem...")
    SDP_problem, varλ, varP = SOS_problem(Δ^2, Δ, upper_bound=upper_bound)
    JuMP.setsolver(SDP_problem, solver)
    info(Base.repr(SDP_problem))

    @time λ, P, ws = solve_SDP(SDP_problem, varλ, varP, ws, solverlog=solverlog)

    return λ, P, ws
end

function saveλandP(name, λ, P, ws)
    save(filename(name, :λ), "λ", λ)
    save(filename(name, :P), "P", P)
    save(filename(name, :warm), "warmstart", ws)
end

Kazhdan(λ::Number,N::Integer) = sqrt(2*λ/N)

function check_property_T(name::String, S, Id, solver, upper_bound, tol, radius, warm::Bool=false)

    isdir(name) || mkdir(name)

    if exists(filename(name, :pm)) && exists(filename(name, :Δ))
        # cached
        Δ = loadLaplacian(name, parent(S[1]))
    else
        # compute
        Δ = computeLaplacian(S, radius)
        save(filename(name, :pm), "pm", parent(Δ).pm)
        save(filename(name, :Δ), "Δ", Δ.coeffs)
    end

    fullpath = joinpath(name, string(upper_bound))
    isdir(fullpath) || mkdir(fullpath)

    files_exist = exists(filename(fullpath, :λ)) && exists(filename(fullpath, :P))

    if !(warm) && files_exist
        λ, P = loadλandP(fullpath)
    else
        warmfile = filename(fullpath, :warm)
        if warm && isfile(warmfile)
            ws = load(warmfile, "warmstart")
        else
            ws = nothing
        end

        λ, P, ws = computeλandP(Δ, upper_bound, solver, ws,
            solverlog=filename(fullpath, :solverlog))
        saveλandP(fullpath, λ, P, ws)
        if λ < 0
            warn("Solver did not produce a valid solution!")
        end
    end

    info("λ = $λ")
    info("sum(P) = $(sum(P))")
    info("maximum(P) = $(maximum(P))")
    info("minimum(P) = $(minimum(P))")

    isapprox(eigvals(P), abs.(eigvals(P)), atol=tol) ||
        warn("The solution matrix doesn't seem to be positive definite!")

    return interpret_results(name, S, radius, λ, P)
end

function interpret_results(name, S, radius, λ, P)

    RG = GroupRing(parent(first(S)), load(filename(name, :pm), "pm"))
    Δ = GroupRingElem(load(filename(name, :Δ), "Δ")[:, 1], RG)
    @time Q = real(sqrtm(Symmetric(P)))

    sgap = distance_to_cone(Δ, λ, Q, wlen=2*radius)

    if sgap > 0
        Kazhdan_κ = Kazhdan(sgap, length(S))
        if Kazhdan_κ > 0
            info("κ($name, S) ≥ $Kazhdan_κ: Group HAS property (T)!")
            return true
        end
    end
    info("λ($name, S) ≥ $sgap < 0: Tells us nothing about property (T)")
    return false
end

include("SDPs.jl")
include("CheckSolution.jl")
include("Orbit-wise.jl")

end # module Property(T)
