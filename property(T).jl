using JuMP
import MathProgBase: AbstractMathProgSolver
import Base: rationalize
using GroupAlgebras

using ProgressMeter
using ValidatedNumerics

function create_product_matrix(basis, limit)
    product_matrix = zeros(Int, (limit,limit))
    basis_dict = Dict{Array, Int}(x => i
        for (i,x) in enumerate(basis))
    for i in 1:limit
        x_inv::eltype(basis) = inv(basis[i])
        for j in 1:limit
            w = x_inv*basis[j]
            product_matrix[i,j] = basis_dict[w]
            # index = findfirst(basis, w)
            # index ≠ 0 || throw(ArgumentError("Product is not supported on basis: $w"))
            # product_matrix[i,j] = index
        end
    end
    return product_matrix
end

function constraints_from_pm(pm, total_length=maximum(pm))
    n = size(pm,1)
    constraints = constraints = [Array{Int,1}[] for x in 1:total_length]
    for j in 1:n
        Threads.@threads for i in 1:n
            idx = pm[i,j]
            push!(constraints[idx], [i,j])
        end
    end
    return constraints
end

function splaplacian_coeff(S, basis, n=length(basis))
    result = spzeros(n)
    result[1] = float(length(S))
    for s in S
        ind = findfirst(basis, s)
        result[ind] += -1.0
    end
    return result
end

function laplacian_coeff(S, basis)
    return full(splaplacian_coeff(S,basis))
end


function create_SDP_problem(matrix_constraints, Δ::GroupAlgebraElement; upper_bound=Inf)
    N = size(Δ.product_matrix,1)
    const Δ² = Δ*Δ
    @assert length(Δ) == length(matrix_constraints)
    m = JuMP.Model();
    JuMP.@variable(m, A[1:N, 1:N], SDP)
    JuMP.@SDconstraint(m, A >= 0)
    JuMP.@constraint(m, sum(A[i] for i in eachindex(A)) == 0)
    JuMP.@variable(m, κ >= 0.0)
    if upper_bound < Inf
        JuMP.@constraint(m, κ <= upper_bound)
    end
    JuMP.@objective(m, Max, κ)

    for (pairs, δ², δ) in zip(matrix_constraints, Δ².coefficients, Δ.coefficients)
        JuMP.@constraint(m, sum(A[i,j] for (i,j) in pairs) == δ² - κ*δ)
    end
    return m
end

function solve_SDP(SDP_problem, solver)
    @show SDP_problem
    @show solver

    JuMP.setsolver(SDP_problem, solver);
    # @time MathProgBase.writeproblem(SDP_problem, "/tmp/SDP_problem")
    solution_status = JuMP.solve(SDP_problem);

    if solution_status != :Optimal
        warn("The solver did not solve the problem successfully!")
    end
    @show solution_status

    κ = JuMP.getvalue(JuMP.getvariable(SDP_problem, :κ))
    A = JuMP.getvalue(JuMP.getvariable(SDP_problem, :A))
    return κ, A
end

function EOI{T<:Number}(Δ::GroupAlgebraElement{T}, κ::T)
    return Δ*Δ - κ*Δ
end

function square_as_elt(vector, elt)
    zzz = zeros(elt.coefficients)
    zzz[1:length(vector)] = vector
#     new_base_elt = GroupAlgebraElement(zzz, elt.product_matrix)
#     return (new_base_elt*new_base_elt).coefficients
    return GroupAlgebras.algebra_multiplication(zzz, zzz, elt.product_matrix)
end

function compute_SOS{T<:Number}(sqrt_matrix::Array{T,2},
                                  elt::GroupAlgebraElement{T})
    n = size(sqrt_matrix,2)
    result = zeros(T, length(elt.coefficients))
    p = Progress(n, 1, "Checking SOS decomposition...", 50)
    for i in 1:n
        result .+= square_as_elt(sqrt_matrix[:,i], elt)
        next!(p)
    end
    return GroupAlgebraElement{T}(result, elt.product_matrix)
end

function correct_to_augmentation_ideal{T<:Rational}(sqrt_matrix::Array{T,2})
    sqrt_corrected = similar(sqrt_matrix)
    l = size(sqrt_matrix,2)
    for i in 1:l
        col = view(sqrt_matrix,:,i)
        sqrt_corrected[:,i] = col - sum(col)//l
        # @assert sum(sqrt_corrected[:,i]) == 0
    end
    return sqrt_corrected
end

function check_solution{T<:Number}(κ::T, sqrt_matrix::Array{T,2}, Δ::GroupAlgebraElement{T}; verbose=true, augmented=false)
    result = compute_SOS(sqrt_matrix, Δ)
    if augmented
        epsilon = GroupAlgebras.ɛ(result)
        if isa(epsilon, Interval)
            @assert 0 in epsilon
        elseif isa(epsilon, Rational)
            @assert epsilon == 0//1
        else
            warn("Does checking for augmentation has meaning for $(typeof(epsilon))?")
        end
    end
    SOS_diff = EOI(Δ, κ) - result

    eoi_SOS_L₁_dist = norm(SOS_diff,1)

    if verbose
        @show κ
        if augmented
            println("ɛ(Δ² - κΔ - ∑ξᵢ*ξᵢ) = ", GroupAlgebras.ɛ(SOS_diff))
        else
            ɛ_dist = GroupAlgebras.ɛ(SOS_diff)
            if typeof(ɛ_dist) <: Interval
                 ɛ_dist = ɛ_dist.lo
            end
            @printf("ɛ(Δ² - κΔ - ∑ξᵢ*ξᵢ) ≈ %.10f\n", ɛ_dist)
        end

        L₁_dist = eoi_SOS_L₁_dist
        if typeof(L₁_dist) <: Interval
            L₁_dist = L₁_dist.lo
        end
        @printf("‖Δ² - κΔ - ∑ξᵢ*ξᵢ‖₁ ≈  %.10f\n", L₁_dist)
    end

    distance_to_cone = κ - 2^3*eoi_SOS_L₁_dist
    return distance_to_cone
end

import ValidatedNumerics.±
function (±)(X::AbstractArray, tol::Real)
    r{T}(x::T) = ( x==zero(T) ? @interval(x) : x ± tol)
    return r.(X)
end

(±)(X::GroupAlgebraElement, tol::Real) = GroupAlgebraElement(X.coefficients ± tol, X.product_matrix)

function Base.rationalize{T<:Integer, S<:Real}(::Type{T},
    X::AbstractArray{S}; tol::Real=eps(eltype(X)))
    r(x) = rationalize(T, x, tol=tol)
    return r.(X)
end

ℚ(x, tol::Real) = rationalize(BigInt, x, tol=tol)


function ℚ_distance_to_positive_cone(Δ::GroupAlgebraElement, κ, A;
    tol=1e-7, verbose=true, rational=false)

    isapprox(eigvals(A), abs(eigvals(A)), atol=tol) ||
        warn("The solution matrix doesn't seem to be positive definite!")
    @assert A == Symmetric(A)
    A_sqrt = real(sqrtm(A))

    # println("")
    # println("Checking in floating-point arithmetic...")
    # @time fp_distance = check_solution(κ, A_sqrt, Δ, verbose=verbose)
    # println("Floating point distance (to positive cone) ≈ $(Float64(trunc(fp_distance,8)))")
    # println("-------------------------------------------------------------")
    # println("")
    #
    # if fp_distance ≤ 0
    #     return fp_distance
    # end

    println("Checking in interval arithmetic...")
    A_sqrtᴵ = A_sqrt ± tol
    κᴵ = κ ± tol
    Δᴵ = Δ ± tol
    @time Interval_distance = check_solution(κᴵ, A_sqrtᴵ, Δᴵ, verbose=verbose)
    # @assert isa(ℚ_distance, Rational)
    println("The actual distance (to positive cone) is contained in $Interval_distance")
    println("-------------------------------------------------------------")
    println("")

    if Interval_distance.lo ≤ 0
        return Interval_distance.lo
    end

    println("Projecting columns of A_sqrt to the augmentation ideal...")
    A_sqrt_ℚ = ℚ(A_sqrt, tol)
    A_sqrt_ℚ_aug = correct_to_augmentation_ideal(A_sqrt_ℚ)
    κ_ℚ = ℚ(κ, tol)
    Δ_ℚ = ℚ(Δ, tol)

    A_sqrt_ℚ_augᴵ = A_sqrt_ℚ_aug ± tol
    κᴵ = κ_ℚ ± tol
    Δᴵ = Δ_ℚ ± tol
    @time Interval_dist_to_Σ² = check_solution(κᴵ, A_sqrt_ℚ_augᴵ, Δᴵ, verbose=verbose, augmented=true)
    println("The Augmentation-projected actual distance (to positive cone) is contained in $Interval_dist_to_Σ²")
    println("-------------------------------------------------------------")
    println("")

    if Interval_dist_to_Σ².lo ≤ 0 || !rational
        return Interval_dist_to_Σ².lo
    else

        println("Checking Projected SOS decomposition in exact rational arithmetic...")
        @time ℚ_dist_to_Σ² = check_solution(κ_ℚ, A_sqrt_ℚ_aug, Δ_ℚ, verbose=verbose, augmented=true)
        @assert isa(ℚ_dist_to_Σ², Rational)
        println("Augmentation-projected rational distance (to positive cone) ≥ $(Float64(trunc(ℚ_dist_to_Σ²,8)))")
        println("-------------------------------------------------------------")
        return ℚ_dist_to_Σ²
    end
end

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
