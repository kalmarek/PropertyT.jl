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

function pmΔfilenames(prefix::String)
    isdir(prefix) || mkdir(prefix)
    pm_filename = joinpath(prefix, "pm.jld")
    Δ_coeff_filename = joinpath(prefix, "delta.jld")
    return pm_filename, Δ_coeff_filename
end

function λSDPfilenames(prefix::String)
    isdir(prefix) || mkdir(prefix)
    λ_filename = joinpath(prefix, "lambda.jld")
    SDP_filename = joinpath(prefix, "SDPmatrix.jld")
    return λ_filename, SDP_filename
end

function ΔandSDPconstraints(prefix::String, G::Group)
    info(LOGGER, "Loading precomputed pm, Δ, sdp_constraints...")
    pm_fname, Δ_fname = pmΔfilenames(prefix)

    product_matrix = load(pm_fname, "pm")
    sdp_constraints = constraints(product_matrix)

    RG = GroupRing(G, product_matrix)
    Δ = GroupRingElem(load(Δ_fname, "Δ")[:, 1], RG)

    return Δ, sdp_constraints
end

function ΔandSDPconstraints{T<:GroupElem}(name::String, S::Vector{T}, Id::T; radius::Int=2)
    info(LOGGER, "Computing pm, Δ, sdp_constraints...")
    Δ, sdp_constraints = ΔandSDPconstraints(S, Id, radius=radius)
    pm_fname, Δ_fname = pmΔfilenames(name)
    save(pm_fname, "pm", parent(Δ).pm)
    save(Δ_fname, "Δ", Δ.coeffs)
    return Δ, sdp_constraints
end

function ΔandSDPconstraints{T<:GroupElem}(S::Vector{T}, Id::T; radius::Int=2)
    info(LOGGER, "Generating balls of sizes $sizes")
    @logtime LOGGER E_R, sizes = Groups.generate_balls(S, Id, radius=2*radius)

    info(LOGGER, "Creating product matrix...")
    @logtime LOGGER pm = GroupRings.create_pm(E_R, GroupRings.reverse_dict(E_R), sizes[radius]; twisted=true)

    info(LOGGER, "Creating sdp_constratints...")
    @logtime LOGGER sdp_constraints = PropertyT.constraints(pm)

    RG = GroupRing(parent(Id), E_R, pm)

    Δ = splaplacian(RG, S)
    return Δ, sdp_constraints
end

function λandP(name::String)
    λ_fname, SDP_fname = λSDPfilenames(name)
    f₁ = exists(λ_fname)
    f₂ = exists(SDP_fname)

    if f₁ && f₂
        info(LOGGER, "Loading precomputed λ, P...")
        λ = load(λ_fname, "λ")
        P = load(SDP_fname, "P")
    else
        throw(ArgumentError("You need to precompute λ and SDP matrix P to load it!"))
    end
    return λ, P
end

function λandP(name::String, SDP_problem::JuMP.Model, varλ, varP, warmstart=false)

    handler = DefaultHandler(
       joinpath(name, "solver_$(string(now())).log"),
       DefaultFormatter("{date}| {msg}")
       )
    handler.levels.x = LOGGER_SOLVER.levels
    LOGGER_SOLVER.handlers["solver_log"] = handler

    if warmstart && isfile(joinpath(name, "warmstart.jld"))
        ws = load(joinpath(name, "warmstart.jld"), "warmstart")
    else
        ws = nothing
    end

    λ, P, warmstart = compute_λandP(SDP_problem, varλ, varP, warmstart=ws)

    delete!(LOGGER_SOLVER.handlers, "solver_log")

    λ_fname, P_fname = λSDPfilenames(name)

    if λ > 0
        save(λ_fname, "λ", λ)
        save(P_fname, "P", P)
        save(joinpath(name, "warmstart.jld"), "warmstart", warmstart)
    else
        throw(ErrorException("Solver did not produce a valid solution!: λ = $λ"))
    end
    return λ, P
end

function fillfrominternal!(m::JuMP.Model, traits)
    # Copied from JuMP/src/solvers.jl:178

    stat::Symbol = MathProgBase.status(m.internalModel)

    numRows, numCols = length(m.linconstr), m.numCols
    m.objBound = NaN
    m.objVal = NaN
    m.colVal = fill(NaN, numCols)
    m.linconstrDuals = Array{Float64}(0)

    discrete = (traits.int || traits.sos)

    if stat == :Optimal
        # If we think dual information might be available, try to get it
        # If not, return an array of the correct length
        if discrete
            m.redCosts = fill(NaN, numCols)
            m.linconstrDuals = fill(NaN, numRows)
        else
            if !traits.conic
                m.redCosts = try
                    MathProgBase.getreducedcosts(m.internalModel)[1:numCols]
                catch
                    fill(NaN, numCols)
                end

                m.linconstrDuals = try
                    MathProgBase.getconstrduals(m.internalModel)[1:numRows]
                catch
                    fill(NaN, numRows)
                end
            elseif !traits.qp && !traits.qc
                JuMP.fillConicDuals(m)
            end
        end
    else
        # Problem was not solved to optimality, attempt to extract useful
        # information anyway

        if traits.lin
            if stat == :Infeasible
                m.linconstrDuals = try
                    infray = MathProgBase.getinfeasibilityray(m.internalModel)
                    @assert length(infray) == numRows
                    infray
                catch
                    suppress_warnings || warn("Infeasibility ray (Farkas proof) not available")
                    fill(NaN, numRows)
                end
            elseif stat == :Unbounded
                m.colVal = try
                    unbdray = MathProgBase.getunboundedray(m.internalModel)
                    @assert length(unbdray) == numCols
                    unbdray
                catch
                    suppress_warnings || warn("Unbounded ray not available")
                    fill(NaN, numCols)
                end
            end
        end
        # conic duals (currently, SOC and SDP only)
        if !discrete && traits.conic && !traits.qp && !traits.qc
            if stat == :Infeasible
                JuMP.fillConicDuals(m)
            end
        end
    end

    # If the problem was solved, or if it terminated prematurely, try
    # to extract a solution anyway. This commonly occurs when a time
    # limit or tolerance is set (:UserLimit)
    if !(stat == :Infeasible || stat == :Unbounded)
        try
            # Do a separate try since getobjval could work while getobjbound does not and vice versa
            objBound = MathProgBase.getobjbound(m.internalModel) + m.obj.aff.constant
            m.objBound = objBound
        end
        try
            objVal = MathProgBase.getobjval(m.internalModel) + m.obj.aff.constant
            colVal = MathProgBase.getsolution(m.internalModel)[1:numCols]
            # Rescale off-diagonal terms of SDP variables
            if traits.sdp
                offdiagvars = JuMP.offdiagsdpvars(m)
                colVal[offdiagvars] /= sqrt(2)
            end
            # Don't corrupt the answers if one of the above two calls fails
            m.objVal = objVal
            m.colVal = colVal
        end
    end

    return stat
end

function compute_λandP(m, varλ, varP; warmstart=nothing)
    λ = 0.0
    P = nothing

    traits = JuMP.ProblemTraits(m, relaxation=false)

    while λ == 0.0
        try
            JuMP.build(m, traits=traits)
            if warmstart != nothing
                p_sol, d_sol, s = warmstart
                MathProgBase.SolverInterface.setwarmstart!(m.internalModel, p_sol; dual_sol = d_sol, slack=s);
            end
            solve_SDP(m)
            λ = MathProgBase.getobjval(m.internalModel)
        catch y
            warn(LOGGER_SOLVER, y)
        end
    end

    warmstart = (m.internalModel.primal_sol, m.internalModel.dual_sol,
          m.internalModel.slack)

    fillfrominternal!(m, traits)

    P = JuMP.getvalue(varP)
    λ = JuMP.getvalue(varλ)

    return λ, P, warmstart
end

Kazhdan_from_sgap(λ,N) = sqrt(2*λ/N)

function check_property_T(name::String, S, Id, solver, upper_bound, tol, radius)

    isdir(name) || mkdir(name)

    if all(exists.(pmΔfilenames(name)))
        # cached
        Δ, sdp_constraints = ΔandSDPconstraints(name, parent(S[1]))
    else
        # compute
        Δ, sdp_constraints = ΔandSDPconstraints(name, S, Id, radius=radius)
    end

    if all(exists.(λSDPfilenames(name)))
        λ, P = λandP(name)
    else
        info(LOGGER, "Creating SDP problem...")
        SDP_problem, λ, P = create_SDP_problem(Δ, sdp_constraints, upper_bound=upper_bound)
        JuMP.setsolver(SDP_problem, solver)

        λ, P = λandP(name, SDP_problem, λ, P)
    end

    info(LOGGER, "λ = $λ")
    info(LOGGER, "sum(P) = $(sum(P))")
    info(LOGGER, "maximum(P) = $(maximum(P))")
    info(LOGGER, "minimum(P) = $(minimum(P))")

    if λ > 0
        pm_fname, Δ_fname = pmΔfilenames(name)
        RG = GroupRing(parent(first(S)), load(pm_fname, "pm"))
        Δ = GroupRingElem(load(Δ_fname, "Δ")[:, 1], RG)

        isapprox(eigvals(P), abs(eigvals(P)), atol=tol) ||
            warn("The solution matrix doesn't seem to be positive definite!")
        @logtime LOGGER Q = real(sqrtm(Symmetric(P)))

        sgap = distance_to_positive_cone(Δ, λ, Q, 2*radius, LOGGER)
        if isa(sgap, Interval)
            sgap = sgap.lo
        end
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
