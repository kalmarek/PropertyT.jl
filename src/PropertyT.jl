module PropertyT

using Nemo
using Groups
using GroupRings

import Nemo: Group, GroupElem, Ring, Generic.perm

using JLD
using JuMP

using Memento

const logger = Memento.config("info", fmt="{msg}")
const solver_logger = Memento.config("info", fmt="{msg}")

function setup_logging(name::String)
   isdir(name) || mkdir(name)

   Memento.add_handler(logger,
      Memento.DefaultHandler(joinpath(name,"full_$(string((now()))).log"),
      Memento.DefaultFormatter("{date}| {msg}")), "full_log")

   e = redirect_stderr(logger.handlers["full_log"].io)

   return logger
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
        esc(info(logger, ts))
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

function exists(fname::String)
   return isfile(fname) || islink(fname)
end

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
    info(logger, "Loading precomputed pm, Δ, sdp_constraints...")
    pm_fname, Δ_fname = pmΔfilenames(prefix)

    product_matrix = load(pm_fname, "pm")
    sdp_constraints = constraints(product_matrix)

    RG = GroupRing(G, product_matrix)
    Δ = GroupRingElem(load(Δ_fname, "Δ")[:, 1], RG)

    return Δ, sdp_constraints
end

function ΔandSDPconstraints{T<:GroupElem}(name::String, S::Vector{T}, Id::T; radius::Int=2)
   info(logger, "Computing pm, Δ, sdp_constraints...")
   Δ, sdp_constraints = ΔandSDPconstraints(S, Id, radius=radius)
   pm_fname, Δ_fname = pmΔfilenames(name)
   save(pm_fname, "pm", parent(Δ).pm)
   save(Δ_fname, "Δ", Δ.coeffs)
   return Δ, sdp_constraints
end

function ΔandSDPconstraints{T<:GroupElem}(S::Vector{T}, Id::T; radius::Int=2)
    info(logger, "Generating balls of sizes $sizes")
    @logtime logger E_R, sizes = Groups.generate_balls(S, Id, radius=2*radius)

    info(logger, "Creating product matrix...")
    @logtime logger pm = GroupRings.create_pm(E_R, GroupRings.reverse_dict(E_R), sizes[radius]; twisted=true)

    info(logger, "Creating sdp_constratints...")
    @logtime logger sdp_constraints = PropertyT.constraints(pm)

    RG = GroupRing(parent(Id), E_R, pm)

    Δ = splaplacian(RG, S)
    return Δ, sdp_constraints
end

function λandP(name::String)
    λ_fname, SDP_fname = λSDPfilenames(name)
    f₁ = exists(λ_fname)
    f₂ = exists(SDP_fname)

    if f₁ && f₂
        info(logger, "Loading precomputed λ, P...")
        λ = load(λ_fname, "λ")
        P = load(SDP_fname, "P")
    else
        throw(ArgumentError("You need to precompute λ and SDP matrix P to load it!"))
    end
    return λ, P
end

function λandP(name::String, SDP_problem::JuMP.Model, varλ, varP)
   add_handler(solver_logger,
      DefaultHandler(joinpath(name, "solver_$(string(now())).log"),
      DefaultFormatter("{date}| {msg}")),
      "solver_log")

   λ, P, warmstart = compute_λandP(SDP_problem, varλ, varP)

   remove_handler(solver_logger, "solver_log")

   λ_fname, P_fname = λSDPfilenames(name)

   if λ > 0
       save(λ_fname, "λ", λ)
       save(P_fname, "P", P)
       @show warmstart[1]
       save(joinpath(name, "warmstart.jld"), "warmstart", warmstart)
   else
       throw(ErrorException("Solver did not produce a valid solution!: λ = $λ"))
   end
   return λ, P

end

function compute_λandP(m, varλ, varP; warmstart=nothing)
    λ = 0.0
    P = nothing
    while λ == 0.0
        try
            if warmstart != nothing
                p_sol, d_sol, s = warmstart
                MathProgBase.SolverInterface.setwarmstart!(m.internalModel, p_sol; dual_sol = d_sol, slack=s);
            end
            solve_SDP(m)
            λ = MathProgBase.getobjval(m.internalModel)
        catch y
            warn(solver_logger, y)
        end
    end
    return λ, P, (p_sol, d_sol, s)
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

    info(logger, "|S| = $(length(S))")
    info(logger, "length(Δ) = $(length(Δ))")
    info(logger, "|R[G]|.pm = $(size(parent(Δ).pm))")

   if all(exists.(λSDPfilenames(name)))
      λ, P = λandP(name)
   else
      info(logger, "Creating SDP problem...")
      SDP_problem, λ, P = create_SDP_problem(Δ, sdp_constraints, upper_bound=upper_bound)
      JuMP.setsolver(SDP_problem, solver)

      λ, P = λandP(name, SDP_problem, λ, P)
   end

   info(logger, "λ = $λ")
   info(logger, "sum(P) = $(sum(P))")
   info(logger, "maximum(P) = $(maximum(P))")
   info(logger, "minimum(P) = $(minimum(P))")

   if λ > 0
      pm_fname, Δ_fname = pmΔfilenames(name)
      RG = GroupRing(parent(first(S)), load(pm_fname, "pm"))
      Δ = GroupRingElem(load(Δ_fname, "Δ")[:, 1], RG)

      isapprox(eigvals(P), abs(eigvals(P)), atol=tol) ||
         warn("The solution matrix doesn't seem to be positive definite!")
     #  @assert P == Symmetric(P)
      @logtime logger Q = real(sqrtm(Symmetric(P)))

      sgap = distance_to_positive_cone(Δ, λ, Q, 2*radius)
      if isa(sgap, Interval)
         sgap = sgap.lo
      end
      if sgap > 0
         info(logger, "λ ≥ $(Float64(trunc(sgap,12)))")
         Kazhdan_κ = Kazhdan_from_sgap(sgap, length(S))
         Kazhdan_κ = Float64(trunc(Kazhdan_κ, 12))
         info(logger, "κ($name, S) ≥ $Kazhdan_κ: Group HAS property (T)!")
         return true
      else
         sgap = Float64(trunc(sgap, 12))
         info(logger, "λ($name, S) ≥ $sgap: Group may NOT HAVE property (T)!")
         return false
      end
   end
   info(logger, "κ($name, S) ≥ $λ < 0: Tells us nothing about property (T)")
   return false
end

include("SDPs.jl")
include("CheckSolution.jl")
include("Orbit-wise.jl")

end # module Property(T)
