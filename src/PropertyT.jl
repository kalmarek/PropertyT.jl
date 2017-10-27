module PropertyT

using Nemo
using Groups
using GroupRings

import Nemo: Group, GroupElem, Ring

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
    sdp_constraints = constraints_from_pm(product_matrix)

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
    B, sizes = Groups.generate_balls(S, Id, radius=2*radius)
    info(logger, "Generated balls of sizes $sizes")

    info(logger, "Creating product matrix...")
    t = @timed pm = GroupRings.create_pm(B, GroupRings.reverse_dict(B), sizes[radius]; twisted=true)
    info(logger, timed_msg(t))

    info(logger, "Creating sdp_constratints...")
    t = @timed sdp_constraints = PropertyT.constraints_from_pm(pm)
    info(logger, timed_msg(t))

    RG = GroupRing(parent(Id), B, pm)

    Δ = splaplacian(RG, S, Id)
    return Δ, sdp_constraints
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
        esc(warn($(esc(logger)), ts))
        val
    end
end

function timed_msg(t)
    elapsed = t[2]
    bytes_alloc = t[3]
    gc_time = t[4]
    gc_diff = t[5]

    return "took: $elapsed s, allocated: $bytes_alloc bytes ($(gc_diff.poolalloc) allocations)."
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
   if exists(joinpath(name, "solver.log"))
       rm(joinpath(name, "solver.log"))
   end

   add_handler(solver_logger,
      DefaultHandler(joinpath(name, "solver_$(string(now())).log"),
      DefaultFormatter("{date}| {msg}")),
      "solver_log")

   λ, P = compute_λandP(SDP_problem, varλ, varP)

   remove_handler(solver_logger, "solver_log")

   λ_fname, P_fname = λSDPfilenames(name)

   if λ > 0
       save(λ_fname, "λ", λ)
       save(P_fname, "P", P)
   else
       throw(ErrorException("Solver did not produce a valid solution!: λ = $λ"))
   end
   return λ, P

end

function compute_λandP(m, varλ, varP)
    λ = 0.0
    P = nothing
    while λ == 0.0
        try
            solve_SDP(m)
            λ = JuMP.getvalue(varλ)
            P = JuMP.getvalue(varP)
        catch y
            warn(solver_logger, y)
        end
    end
    return λ, P
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
      # cached
      λ, P = λandP(name)
   else
      # compute
      info(logger, "Creating SDP problem...")

      t = @timed SDP_problem, λ, P = create_SDP_problem(Δ, sdp_constraints, upper_bound=upper_bound)
      info(logger, timed_msg(t))

      JuMP.setsolver(SDP_problem, solver)

      λ, P = λandP(name, SDP_problem, λ, P)
   end

   info(logger, "λ = $λ")
   info(logger, "sum(P) = $(sum(P))")
   info(logger, "maximum(P) = $(maximum(P))")
   info(logger, "minimum(P) = $(minimum(P))")

   if λ > 0

      isapprox(eigvals(P), abs(eigvals(P)), atol=tol) ||
         warn("The solution matrix doesn't seem to be positive definite!")
     #  @assert P == Symmetric(P)
      Q = real(sqrtm(Symmetric(P)))

      sgap = check_distance_to_positive_cone(Δ, λ, Q, 2*radius, tol=tol)
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
