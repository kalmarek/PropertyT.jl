using JuMP
import Base: rationalize
using GroupAlgebras

function products{T}(U::AbstractVector{T}, V::AbstractVector{T})
    result = Vector{T}()
    for u in U
        for v in V
            push!(result, u*v)
        end
    end
    return unique(result)
end

function create_product_matrix(basis, limit)
    product_matrix = zeros(Int, (limit,limit))
    for i in 1:limit
        x_inv::eltype(basis) = inv(basis[i])
        for j in 1:limit
            w = x_inv*basis[j]
            index = findfirst(basis, w)
            index ≠ 0 || throw(ArgumentError("Product is not supported on basis: $w"))
            product_matrix[i,j] = index
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
    result[1] = length(S)
    for s in S
        ind = findfirst(basis, s)
        result[ind] += -1
    end
    return result
end

function laplacian_coeff(S, basis)
    return full(splaplacian_coeff(S,basis))
end


function create_SDP_problem(matrix_constraints, Δ::GroupAlgebraElement)
    N = size(Δ.product_matrix,1)
    const Δ² = Δ*Δ
    @assert length(Δ) == length(matrix_constraints)
    m = JuMP.Model();
    JuMP.@variable(m, A[1:N, 1:N], SDP)
    JuMP.@SDconstraint(m, A >= zeros(size(A)))
    JuMP.@variable(m, κ >= 0.0)
    JuMP.@objective(m, Max, κ)

    for (pairs, δ², δ) in zip(matrix_constraints, Δ².coefficients, Δ.coefficients)
        JuMP.@constraint(m, sum(A[i,j] for (i,j) in pairs) == δ² - κ*δ)
    end
    return m
end

function solve_SDP(sdp_constraints, Δ, solver; verbose=true)
    SDP_problem = create_SDP_problem(sdp_constraints, Δ);
    verbose && @show solver

    JuMP.setsolver(SDP_problem, solver);
    verbose && @show SDP_problem
    # @time MathProgBase.writeproblem(SDP_problem, "/tmp/SDP_problem")
    solution_status = JuMP.solve(SDP_problem);
    verbose && @show solution_status

    if solution_status != :Optimal
        throw(ExceptionError("The solver did not solve the problem successfully!"))
    else
        κ = SDP_problem.objVal;
        A = JuMP.getvalue(JuMP.getvariable(SDP_problem, :A));;
    end
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
    # result = zeros(T, length(elt.coefficients))
    result = @parallel (+) for i in 1:n
        square_as_elt(sqrt_matrix[:,i], elt)
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

function check_solution(κ, sqrt_matrix, Δ; verbose=true, augmented=false)
    result = compute_SOS(sqrt_matrix, Δ)
    if augmented
        @assert GroupAlgebras.ɛ(result) == 0//1
    end
    SOS_diff = EOI(Δ, κ) - result

    eoi_SOS_L₁_dist = norm(SOS_diff,1)

    if verbose
        @show κ
        if augmented
            println("ɛ(Δ² - κΔ - ∑ξᵢ*ξᵢ) = ", GroupAlgebras.ɛ(SOS_diff))
        else
            ɛ_dist = Float64(round(GroupAlgebras.ɛ(SOS_diff),12))
            println("ɛ(Δ² - κΔ - ∑ξᵢ*ξᵢ) ≈ $ɛ_dist")
        end
        L₁_dist = Float64(round(eoi_SOS_L₁_dist, 12))
        println("‖Δ² - κΔ - ∑ξᵢ*ξᵢ‖₁ ≈ $L₁_dist")
    end

    distance_to_cone = 2^3*eoi_SOS_L₁_dist - κ
    return distance_to_cone
end

function rationalize{T<:Integer, S<:Real}(::Type{T},
    X::AbstractArray{S}; tol::Real=eps(eltype(X)))
    r(x) = rationalize(T, x, tol=tol)
    return r.(X)
end;

ℚ(x, tol::Real) = rationalize(BigInt, x, tol=tol)


function ℚ_distance_to_positive_cone(Δ::GroupAlgebraElement,
    κ::Float64,
    A::Array{Float64,2};
    tol=10.0^-7,
    verbose=true)

    @show maximum(A)
    if maximum(A) < 1e-2
        warn("Solver might not solved the problem successfully and the positive solution is due to floating-point error, proceeding anyway...")
    end

    @assert isapprox(eigvals(A), abs(eigvals(A)), atol=TOL)
    @assert A == Symmetric(A)
    A_sqrt = real(sqrtm(A))
    println("")

    println("Checking in floating-point arithmetic...")
    fp_distance = check_solution(κ, A_sqrt, Δ, verbose=VERBOSE)
    println("Distance to positive cone ≈ $(Float64(trunc(fp_distance,8)))")
    println("-------------------------------------------------------------")
    println("")

    println("Checking in rational arithmetic...")
    κ_ℚ = ℚ(trunc(κ,Int(abs(log10(tol)))), TOL)
    @assert κ - κ_ℚ ≥ 0
    A_sqrt_ℚ, Δ_ℚ = ℚ(A_sqrt, TOL), ℚ(Δ, TOL)
    ℚ_distance = check_solution(κ_ℚ, A_sqrt_ℚ, Δ_ℚ, verbose=VERBOSE)
    @assert isa(ℚ_distance, Rational)
    println("Distance to positive cone ≈ $(Float64(trunc(ℚ_distance,8)))")
    println("-------------------------------------------------------------")
    println("")

    println("Projecting columns of A_sqrt to the augmentation ideal...")
    A_sqrt_ℚ_aug = correct_to_augmentation_ideal(A_sqrt_ℚ)
    ℚ_dist_to_Σ² = check_solution(κ_ℚ, A_sqrt_ℚ_aug, Δ_ℚ, verbose=VERBOSE, augmented=true)
    @assert isa(ℚ_dist_to_Σ², Rational)
    s = (ℚ_dist_to_Σ² < 0? "≤": "≥")
    println("Distance to positive cone $s $(Float64(trunc(ℚ_dist_to_Σ²,8)))")

    return ℚ_dist_to_Σ²
end
