module PropertyT

using GroupAlgebras
import SCS.SCSSolver

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
        println("Loading precomputed pm, Δ, sdp_constraints...")
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
    Δ, sdp_constraints = ΔandSDPconstraints(ID, generating_func())
    save(pm_fname, "pm", Δ.product_matrix)
    save(Δ_fname, "Δ", Δ.coefficients)
    return Δ, sdp_constraints
end

function κandA(name::String)
    κ_fname, SDP_fname = κSDPfilenames(name)
    f₁ = isfile(κ_fname)
    f₂ = isfile(SDP_fname)
    if f₁ && f₂
        println("Loading precomputed κ, A...")
        κ = load(κ_fname, "κ")
        A = load(SDP_fname, "A")
    else
        throw(ArgumentError("You need to precompute κ and SDP matrix A to load it!"))
    end
    return κ, A
end

function κandA(name::String, sdp_constraints, Δ::GroupAlgebraElement, solver::AbstractMathProgSolver; upper_bound=Inf)
    println("Creating SDP problem...")
    @time SDP_problem = create_SDP_problem(sdp_constraints, Δ; upper_bound=upper_bound)
    println("Solving SDP problem maximizing κ...")
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
    verbose=true, tol=1e-6, upper_bound=Inf)

    # solver = MosekSolver(INTPNT_CO_TOL_REL_GAP=tol, QUIET=!verbose)
    solver = SCSSolver(eps=tol, max_iters=100000, verbose=verbose)

    @show name
    @show verbose
    @show tol


    Δ, sdp_constraints = try
        ΔandSDPconstraints(name)
    catch err
        if isa(err, ArgumentError)
            ΔandSDPconstraints(name, ID, generate_B₄)
        else
            throw(err)
        end
    end
    println("|S| = $(countnz(Δ.coefficients) -1)")
    @show length(Δ)
    @show size(Δ.product_matrix)

    κ, A = try
        κandA(name)
    catch err
        if isa(err, ArgumentError)
            κandA(name, sdp_constraints, Δ, solver; upper_bound=upper_bound)
        else
            throw(err)
        end
    end

    @show κ
    @show sum(A)
    @show maximum(A)
    @show minimum(A)

    if κ > 0

        true_kappa = ℚ_distance_to_positive_cone(Δ, κ, A, tol=tol, verbose=verbose, rational=true)
        true_kappa = Float64(trunc(true_kappa,12))
        if true_kappa > 0
            println("κ($name, S) ≥ $true_kappa: Group HAS property (T)!")
        else
            println("κ($name, S) ≥ $true_kappa: Group may NOT HAVE property (T)!")
        end
    else
        println("κ($name, S) ≥ $κ < 0: Tells us nothing about property (T)")
    end
end

end # module Property(T)
