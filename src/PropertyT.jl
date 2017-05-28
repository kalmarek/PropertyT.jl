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
    if f₁ && f₂
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

function timed_msg(t)
    elapsed = t[2]
    bytes_alloc = t[3]
    gc_time = t[4]
    gc_diff = t[5]

    return "took: $elapsed s, allocated: $bytes_alloc bytes ($(gc_diff.poolalloc) allocations)."
end


function λandP(name::String, opts...)
    try
        return λandP(name)
    catch err
        if isa(err, ArgumentError)
            if isfile(joinpath(name, "solver.log"))
                rm(joinpath(name, "solver.log"))
            end

            add_handler(solver_logger, DefaultHandler(joinpath(name, "solver.log"), DefaultFormatter("{date}| {msg}")), "solver_log")

            info(logger, "Creating SDP problem...")

            λ, P = compute_λandP(opts...)

            remove_handler(solver_logger, "solver_log")

            λ_fname, P_fname = λSDPfilenames(name)

            if λ > 0
                save(λ_fname, "λ", λ)
                save(P_fname, "P", P)
            else
                throw(ErrorException("Solver $solver did not produce a valid solution!: λ = $λ"))
            end
            return λ, P

        else
            # throw(err)
            error(logger, err)
        end
    end
end

function compute_λandP(sdp_constraints, Δ::GroupRingElem, solver::AbstractMathProgSolver, upper_bound=Inf)

    t = @timed SDP_problem = create_SDP_problem(sdp_constraints, Δ; upper_bound=upper_bound)
    info(logger, timed_msg(t))

    λ = 0.0
    P = nothing
    while λ == 0.0
        try
            λ, P = solve_SDP(SDP_problem, solver)
        catch y
            warn(solver_logger, y)
        end
    end
    return λ, P
end

Kazhdan_from_sgap(λ,N) = sqrt(2*λ/N)

function check_property_T(name::String, generating_set::Function,
    solver, upper_bound, tol, radius)

    if !isdir(name)
        mkdir(name)
    end

    add_handler(logger, DefaultHandler("./$name/full.log", DefaultFormatter("{date}| {msg}")), "full_log")
    e = redirect_stderr(logger.handlers["full_log"].io)
    info(logger, "Group:       $name")
    info(logger, "Precision:   $tol")
    info(logger, "Upper bound: $upper_bound")

    Δ, sdp_constraints = ΔandSDPconstraints(name, generating_set, radius)

    S = countnz(Δ.coeffs) - 1
    info(logger, "|S| = $S")
    info(logger, "length(Δ) = $(length(Δ))")
    info(logger, "|R(G)|.pm = $(size(parent(Δ).pm))")

    λ, P = λandP(name, sdp_constraints, Δ, solver, upper_bound)

    info(logger, "λ = $λ")
    info(logger, "sum(P) = $(sum(P))")
    info(logger, "maximum(P) = $(maximum(P))")
    info(logger, "minimum(P) = $(minimum(P))")

    if λ > 0
        spectral_gap = check_distance_to_positive_cone(Δ, λ, P, tol=tol, rational=false)
        if isa(spectral_gap, Interval)
            sgap = spectral_gap.lo
        end
        if sgap > 0
            info(logger, "λ ≥ $(Float64(trunc(sgap,12)))")
                Kazhdan_κ = Kazhdan_from_sgap(sgap, S)
                Kazhdan_κ = Float64(trunc(Kazhdan_κ, 12))
                info(logger, "κ($name, S) ≥ $Kazhdan_κ: Group HAS property (T)!")

        else
            sgap = Float64(trunc(sgap, 12))
            info(logger, "λ($name, S) ≥ $sgap: Group may NOT HAVE property (T)!")
        end
    else
        info(logger, "κ($name, S) ≥ $λ < 0: Tells us nothing about property (T)")
    end
end

end # module Property(T)
