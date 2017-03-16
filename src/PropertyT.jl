module PropertyT

using JLD
using GroupAlgebras
import SCS.SCSSolver
using Memento

const logger = basic_config("info", fmt="{msg}")
const solver_logger = basic_config("info")

include("sdps.jl")
include("checksolution.jl")

function pmΔfilenames(name::String)
    if !isdir(name)
        mkdir(name)
    end
    prefix = name
    pm_filename = joinpath(prefix, "product_matrix.jld")
    Δ_coeff_filename = joinpath(prefix, "delta.coefficients.jld")
    return pm_filename, Δ_coeff_filename
end

function κSDPfilenames(name::String)
    if !isdir(name)
        mkdir(name)
    end
    prefix = name
    κ_filename = joinpath(prefix, "kappa.jld")
    SDP_filename = joinpath(prefix, "SDPmatrixA.jld")
    return κ_filename, SDP_filename
end

function ΔandSDPconstraints(name::String)
    pm_fname, Δ_fname = pmΔfilenames(name)
    f₁ = isfile(pm_fname)
    f₂ = isfile(Δ_fname)
    if f₁ && f₂
        info(logger, "Loading precomputed pm, Δ, sdp_constraints...")
        product_matrix = load(pm_fname, "pm")
        L = load(Δ_fname, "Δ")[:, 1]
        Δ = GroupAlgebraElement(L, Array{Int,2}(product_matrix))
        sdp_constraints = constraints_from_pm(product_matrix)
    else
        throw(ArgumentError("You need to precompute pm and Δ to load it!"))
    end
    return Δ, sdp_constraints
end

function ΔandSDPconstraints(name::String, ID, generating_func::Function)
    pm_fname, Δ_fname = pmΔfilenames(name)
    Δ, sdp_constraints = Main.ΔandSDPconstraints(ID, generating_func())
    save(pm_fname, "pm", Δ.product_matrix)
    save(Δ_fname, "Δ", Δ.coefficients)
    return Δ, sdp_constraints
end

function κandA(name::String)
    κ_fname, SDP_fname = κSDPfilenames(name)
    f₁ = isfile(κ_fname)
    f₂ = isfile(SDP_fname)
    if f₁ && f₂
        info(logger, "Loading precomputed κ, A...")
        κ = load(κ_fname, "κ")
        A = load(SDP_fname, "A")
    else
        throw(ArgumentError("You need to precompute κ and SDP matrix A to load it!"))
    end
    return κ, A
end

function timed_msg(t)
    elapsed = t[2]
    bytes_alloc = t[3]
    gc_time = t[4]
    gc_diff = t[5]

    return "took: $elapsed s, allocated: $bytes_alloc bytes ($(gc_diff.poolalloc) allocations)."
end

function κandA(name::String, sdp_constraints, Δ::GroupAlgebraElement, solver::AbstractMathProgSolver; upper_bound=Inf)
    if isfile("$name/solver.log")
        rm("$name/solver.log")
    end

    add_handler(solver_logger, DefaultHandler("./$name/solver.log", DefaultFormatter("{date}| {msg}")), "solver")

    info(logger, "Creating SDP problem...")
    t = @timed SDP_problem = create_SDP_problem(sdp_constraints, Δ; upper_bound=upper_bound)
    info(logger, timed_msg(t))

    κ, A = solve_SDP(SDP_problem, solver)
    κ_fname, A_fname = κSDPfilenames(name)
    if κ > 0
        save(κ_fname, "κ", κ)
        save(A_fname, "A", A)
    else
        throw(ErrorException("Solver $solver did not produce a valid solution!: κ = $κ"))
    end
    return κ, A
end

function check_property_T(name::String, ID, generate_B₄::Function;
    tol=1e-6, upper_bound=Inf)

    if !isdir(name)
        mkdir(name)
    end

    add_handler(logger, DefaultHandler("./$name/full.log", DefaultFormatter("{date}| {msg}")), "full")
    info(logger, "Group: $name")
    info(logger, "Precision: $tol")

    # solver = MosekSolver(INTPNT_CO_TOL_REL_GAP=tol, QUIET=false)
    solver = SCSSolver(eps=tol, max_iters=1000000, verbose=true)

    Δ, sdp_constraints = try
        ΔandSDPconstraints(name)
    catch err
        if isa(err, ArgumentError)
            ΔandSDPconstraints(name, ID, generate_B₄)
        else
            error(logger, err)
        end
    end
    S = countnz(Δ.coefficients) - 1
    info(logger, "|S| = $S")
    info(logger, "length(Δ) = $(length(Δ))")
    info(logger, "size(Δ.product_matrix) = $(size(Δ.product_matrix))")

    κ, A = try
        κandA(name)
    catch err
        if isa(err, ArgumentError)
            κandA(name, sdp_constraints, Δ, solver; upper_bound=upper_bound)
        else
            # throw(err)
            error(logger, err)
        end
    end

    info(logger, "κ = $κ")
    info(logger, "sum(A) = $(sum(A))")
    info(logger, "maximum(A) = $(maximum(A))")
    info(logger, "minimum(A) = $(minimum(A))")

    if κ > 0
        spectral_gap = check_distance_to_positive_cone(Δ, κ, A, tol=tol, rational=false)
        Kazhdan_κ = sqrt(2*spectral_gap/S)
        Kazhdan_κ = Float64(trunc(Kazhdan_κ,12))
        if true_kappa > 0
            info(logger, "κ($name, S) ≥ $Kazhdan_κ: Group HAS property (T)!")
        else
            info(logger, "κ($name, S) ≥ $Kazhdan_κ: Group may NOT HAVE property (T)!")
        end
    else
        info(logger, "κ($name, S) ≥ $κ < 0: Tells us nothing about property (T)")
    end
end

end # module Property(T)
