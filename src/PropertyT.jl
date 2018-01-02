module PropertyT

using Nemo
using Groups
using GroupRings

import Nemo: Group, GroupElem, Ring, Generic.perm

using JLD
using JuMP
using MathProgBase

using Memento

const LOGGER = Memento.config("info", fmt="{msg}")
const LOGGER_SOLVER = Memento.config("info", fmt="{msg}")

function setup_logging(name::String)
    isdir(name) || mkdir(name)

    handler = Memento.DefaultHandler(
        joinpath(name,"full_$(string((now()))).log"),    Memento.DefaultFormatter("{date}| {msg}")
    )
    handler.levels.x = LOGGER.levels
    LOGGER.handlers["full_log"] = handler

    e = redirect_stderr(logger.handlers["full_log"].io)

    return LOGGER
end

macro logtime(logger, ex)
    quote
        local stats = Base.gc_num()
        local elapsedtime = Base.time_ns()
        local val = $(esc(ex))
        elapsedtime = Base.time_ns() - elapsedtime
        local diff = Base.GC_Diff(Base.gc_num(), stats)
        local ts = time_string(elapsedtime, diff.allocd, diff.total_time,
                               Base.gc_alloc_count(diff))
        $(esc(info))($(esc(logger)), ts)
        val
    end
end

function time_string(elapsedtime, bytes, gctime, allocs)
    str = @sprintf("%10.6f seconds", elapsedtime/1e9)
    if bytes != 0 || allocs != 0
        bytes, mb = Base.prettyprint_getunits(bytes, length(Base._mem_units), Int64(1024))
        allocs, ma = Base.prettyprint_getunits(allocs, length(Base._cnt_units), Int64(1000))
        if ma == 1
            str*= @sprintf(" (%d%s allocation%s: ", allocs, Base._cnt_units[ma], allocs==1 ? "" : "s")
        else
            str*= @sprintf(" (%.2f%s allocations: ", allocs, Base._cnt_units[ma])
        end
        if mb == 1
            str*= @sprintf("%d %s%s", bytes, Base._mem_units[mb], bytes==1 ? "" : "s")
        else
            str*= @sprintf("%.3f %s", bytes, Base._mem_units[mb])
        end
        if gctime > 0
            str*= @sprintf(", %.2f%% gc time", 100*gctime/elapsedtime)
        end
        str*=")"
    elseif gctime > 0
        str*= @sprintf(", %.2f%% gc time", 100*gctime/elapsedtime)
    end
    return str
end

exists(fname::String) = isfile(fname) || islink(fname)

function filename(prefix, s::Symbol)
    isdir(prefix) || mkdir(prefix)
    return filename(prefix, Val{s})
end

filename(prefix::String, ::Type{Val{:pm}}) = joinpath(prefix, "pm.jld")
filename(prefix::String, ::Type{Val{:Δ}})  = joinpath(prefix, "delta.jld")
filename(prefix::String, ::Type{Val{:λ}})  = joinpath(prefix, "lambda.jld")
filename(prefix::String, ::Type{Val{:P}})  = joinpath(prefix, "SDPmatrix.jld")

function Laplacian(name::String, G::Group)
    info(LOGGER, "Loading precomputed Δ...")
    if exists(filename(name, :Δ)) && exists(filename(name, :pm))
        RG = GroupRing(G, load(filename(name, :pm), "pm"))
        Δ = GroupRingElem(load(filename(name, :Δ), "Δ")[:, 1], RG)
    else
        throw("You need to precompute $(filename(name, :pm)) and $(filename(name, :Δ)) to load it!")
    end
    return Δ
end

function Laplacian{T<:GroupElem}(name::String, S::Vector{T}, Id::T; radius::Int=2)
    info(LOGGER, "Computing multiplication table, Δ...")
    Δ = Laplacian(S, Id, radius=radius)
    save(filename(name, :pm), "pm", parent(Δ).pm)
    save(filename(name, :Δ), "Δ", Δ.coeffs)
    return Δ
end

function Laplacian{T<:GroupElem}(S::Vector{T}, Id::T; radius::Int=2)
    info(LOGGER, "Generating metric ball of radius $radius...")
    @logtime LOGGER E_R, sizes = Groups.generate_balls(S, Id, radius=2*radius)
    info(LOGGER, "Generated balls of sizes $sizes.")

    info(LOGGER, "Creating product matrix...")
    @logtime LOGGER pm = GroupRings.create_pm(E_R, GroupRings.reverse_dict(E_R), sizes[radius]; twisted=true)

    RG = GroupRing(parent(Id), E_R, pm)

    Δ = splaplacian(RG, S)
    return Δ
end

function λandP(name::String)
    λ_fname = filename(name, :λ)
    P_fname = filename(name, :P)

    if exists(λ_fname) && exists(P_fname)
        λ = load(λ_fname, "λ")
        P = load(P_fname, "P")
    else
        throw("You need to precompute $λ_fname and $P_fname to load it!")
    end
    return λ, P
end

function λandP(name::String, SDP::JuMP.Model, varλ, varP, warmstart=true)

    if warmstart && isfile(filename(name, :warm))
        ws = load(filename(name, :warm), "warmstart")
    else
        ws = nothing
    end

    solver_log = solverlogger(name)

    Base.Libc.flush_cstdio()
    o = redirect_stdout(solver_log.handlers["solver_log"].io)
    Base.Libc.flush_cstdio()

    λ, P, warmstart = solve_SDP(SDP, varλ, varP, warmstart=ws)

    Base.Libc.flush_cstdio()
    redirect_stdout(o)

    delete!(solver_log.handlers, "solver_log")

    if λ > 0
        save(filename(name, :λ), "λ", λ)
        save(filename(name, :P), "P", P)
        save(filename(name, :warm), "warmstart", warmstart)
    else
        throw(ErrorException("Solver did not produce a valid solution: λ = $λ"))
    end
    return λ, P
end

Kazhdan_from_sgap(λ,N) = sqrt(2*λ/N)

function check_property_T(name::String, S, Id, solver, upper_bound, tol, radius)

    isdir(name) || mkdir(name)

    if exists(filename(name, :pm)) && exists(filename(name, :Δ))
        # cached
        Δ = Laplacian(name, parent(S[1]))
    else
        # compute
        Δ = Laplacian(name, S, Id, radius=radius)
    end

    if exists(filename(name, :λ)) && exists(filename(name, :P))
        info(LOGGER, "Loading precomputed λ, P...")
        λ, P = λandP(name)
    else
        info(LOGGER, "Creating SDP problem...")
        SDP_problem, λ_var, P_var = create_SDP_problem(Δ, PropertyT.constraints(parent(Δ).pm), upper_bound=upper_bound)
        JuMP.setsolver(SDP_problem, solver)

        λ, P = λandP(name, SDP_problem, λ_var, P_var)
    end

    info(LOGGER, "λ = $λ")
    info(LOGGER, "sum(P) = $(sum(P))")
    info(LOGGER, "maximum(P) = $(maximum(P))")
    info(LOGGER, "minimum(P) = $(minimum(P))")

    if λ > 0

        RG = GroupRing(parent(first(S)), load(filename(name, :pm), "pm"))
        Δ = GroupRingElem(load(filename(name, :Δ), "Δ")[:, 1], RG)

        isapprox(eigvals(P), abs(eigvals(P)), atol=tol) ||
            warn("The solution matrix doesn't seem to be positive definite!")
        @logtime LOGGER Q = real(sqrtm(Symmetric(P)))

        sgap = check_distance_to_cone(Δ, λ, Q, 2*radius, LOGGER)

        if sgap > 0
            info(LOGGER, "λ ≥ $(Float64(trunc(sgap,12)))")
            Kazhdan_κ = Kazhdan_from_sgap(sgap, length(S))
            Kazhdan_κ = Float64(trunc(Kazhdan_κ, 12))
            info(LOGGER, "κ($name, S) ≥ $Kazhdan_κ: Group HAS property (T)!")
            return true
        else
            sgap = Float64(trunc(sgap, 12))
            info(LOGGER, "λ($name, S) ≥ $sgap: Group may NOT HAVE property (T)!")
            return false
        end
    end
    info(LOGGER, "κ($name, S) ≥ $λ < 0: Tells us nothing about property (T)")
    return false
end

include("SDPs.jl")
include("CheckSolution.jl")
include("Orbit-wise.jl")

end # module Property(T)
