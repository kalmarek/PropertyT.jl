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

function ΔandSDPconstraints(name::String, G::Group)
    info(logger, "Loading precomputed pm, Δ, sdp_constraints...")
    pm_fname, Δ_fname = pmΔfilenames(name)

    product_matrix = load(pm_fname, "pm")
    sdp_constraints = constraints_from_pm(product_matrix)

    RG = GroupRing(G, product_matrix)
    Δ = GroupRingElem(load(Δ_fname, "Δ")[:, 1], RG)

    return Δ, sdp_constraints
end

function ΔandSDPconstraints{T<:GroupElem}(name::String, S::Vector{T}, radius::Int)
   S, Id = generating_set()
   info(logger, "Computing pm, Δ, sdp_constraints...")
   t = @timed Δ, sdp_constraints = ΔandSDPconstraints(S, radius)
   info(logger, timed_msg(t))
   pm_fname, Δ_fname = pmΔfilenames(name)
   save(pm_fname, "pm", parent(Δ).pm)
   save(Δ_fname, "Δ", Δ.coeffs)
end

function ΔandSDPconstraints{T<:GroupElem}(S::Vector{T}, r::Int=2)
    Id = parent(S[1])()
    B, sizes = Groups.generate_balls(S, Id, radius=2*r)
    info(logger, "Generated balls of sizes $sizes")

    info(logger, "Creating product matrix...")
    t = @timed pm = GroupRings.create_pm(B, GroupRings.reverse_dict(B), sizes[r]; twisted=true)
    info(logger, timed_msg(t))

    info(logger, "Creating sdp_constratints...")
    t = @timed sdp_constraints = PropertyT.constraints_from_pm(pm)
    info(logger, timed_msg(t))

    RG = GroupRing(parent(Id), B, pm)

    Δ = splaplacian(RG, S, B[1:sizes[r]], sizes[2*r])
    return Δ, sdp_constraints
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

   Memento.add_handler(logger,
      Memento.DefaultHandler(joinpath(name,"full_$(string((now()))).log"),
      Memento.DefaultFormatter("{date}| {msg}")),
      "full_log")

   e = redirect_stderr(logger.handlers["full_log"].io)

   return logger
end


function check_property_T(name::String, generating_set,
    solver, upper_bound, tol, radius)

    if !isdir(name)
        mkdir(name)
    end

    setup_logging(name)

    if all(isfile.(pmΔfilenames(name)))
        # cached
        Δ, sdp_constraints = ΔandSDPconstraints(name, parent(S[1]))
    else
        # compute
        Δ, sdp_constraints = ΔandSDPconstraints(name, S, radius)
    end

    S = countnz(Δ.coeffs) - 1
    info(logger, "|S| = $S")
    info(logger, "length(Δ) = $(length(Δ))")
    info(logger, "|R(G)|.pm = $(size(parent(Δ).pm))")

   if all(isfile.(λandP(name)))
      λ, P = λandP(name)
   else
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
