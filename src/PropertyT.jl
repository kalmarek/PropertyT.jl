module PropertyT

using JLD
using GroupRings
using Memento

const logger = Memento.config("info", fmt="{msg}")
const solver_logger = Memento.config("info", fmt="{msg}")

include("sdps.jl")
include("checksolution.jl")

function pmΔfilenames(name::String)
    if !isdir(name)
        mkdir(name)
    end
    prefix = name
    pm_filename = joinpath(prefix, "product_matrix.jld")
    Δ_coeff_filename = joinpath(prefix, "delta.coeffs.jld")
    return pm_filename, Δ_coeff_filename
end

function λSDPfilenames(name::String)
    if !isdir(name)
        mkdir(name)
    end
    prefix = name
    λ_filename = joinpath(prefix, "lambda.jld")
    SDP_filename = joinpath(prefix, "SDPmatrix.jld")
    return λ_filename, SDP_filename
end

function ΔandSDPconstraints(name::String)
    pm_fname, Δ_fname = pmΔfilenames(name)
    f₁ = isfile(pm_fname)
    f₂ = isfile(Δ_fname)
    if f₁ && f₂ && false
        info(logger, "Loading precomputed pm, Δ, sdp_constraints...")
        product_matrix = load(pm_fname, "pm")
        L = load(Δ_fname, "Δ")[:, 1]
        Δ = GroupRingElem(L, Array{Int,2}(product_matrix))
        sdp_constraints = constraints_from_pm(product_matrix)
    else
        throw(ArgumentError("You need to precompute pm and Δ to load it!"))
    end
    return Δ, sdp_constraints
end

function ΔandSDPconstraints(name::String, generating_set::Function, radius::Int)
    try
        return ΔandSDPconstraints(name)
    catch err
        if isa(err, ArgumentError)
            pm_fname, Δ_fname = pmΔfilenames(name)
            S, Id = generating_set()
            info(logger, "Computing pm, Δ, sdp_constraints...")
            t = @timed Δ, sdp_constraints = Main.ΔandSDPconstraints(Id, S, radius)
            info(logger, timed_msg(t))

            save(pm_fname, "pm", parent(Δ).pm)
            save(Δ_fname, "Δ", Δ.coeffs)
            return Δ, sdp_constraints
        else
            error(logger, err)
        end
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
    f₁ = isfile(λ_fname)
    f₂ = isfile(SDP_fname)

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
   if isfile(joinpath(name, "solver.log"))
       rm(joinpath(name, "solver.log"))
   end

   add_handler(solver_logger, DefaultHandler(joinpath(name, "solver.log"), DefaultFormatter("{date}| {msg}")), "solver_log")

   λ, P = compute_λandP(SDP_problem, varλ, varP)

   remove_handler(solver_logger, "solver_log")

   λ_fname, P_fname = λSDPfilenames(name)

   if λ > 0
       save(λ_fname, "λ", λ)
       save(P_fname, "P", P)
   else
       throw(ErrorException("Solver $solver did not produce a valid solution!: λ = $λ"))
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

function setup_logging(name::String)

   Memento.add_handler(logger, Memento.DefaultHandler(joinpath(name,"full.log"), Memento.DefaultFormatter("{date}| {msg}")), "full_log")

   e = redirect_stderr(logger.handlers["full_log"].io)

   return logger
end


function check_property_T(name::String, generating_set::Function,
    solver, upper_bound, tol, radius)

    if !isdir(name)
        mkdir(name)
    end

    setup_logging(name)

    Δ, sdp_constraints = ΔandSDPconstraints(name, generating_set, radius)

    S = countnz(Δ.coeffs) - 1
    info(logger, "|S| = $S")
    info(logger, "length(Δ) = $(length(Δ))")
    info(logger, "|R(G)|.pm = $(size(parent(Δ).pm))")

   λ, P = try
      λandP(name)
   catch err
      if isa(err, ArgumentError)
         info(logger, "Creating SDP problem...")

         t = @timed SDP_problem, λ, P = create_SDP_problem(Δ, sdp_constraints, upper_bound=upper_bound)
         info(logger, timed_msg(t))

         JuMP.setsolver(SDP_problem, solver)

         λandP(name, SDP_problem, λ, P)
      end
   end

   info(logger, "λ = $λ")
   info(logger, "sum(P) = $(sum(P))")
   info(logger, "maximum(P) = $(maximum(P))")
   info(logger, "minimum(P) = $(minimum(P))")

   if λ > 0
      sgap = check_distance_to_positive_cone(Δ, λ, P, tol=tol, rational=false, len=2*radius)
      if isa(sgap, Interval)
           sgap = sgap.lo
      end
      if sgap > 0
           info(logger, "λ ≥ $(Float64(trunc(sgap,12)))")
            Kazhdan_κ = Kazhdan_from_sgap(sgap, S)
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

end # module Property(T)
