using ProgressMeter
using ValidatedNumerics
import Base: rationalize

function EOI{T<:Number}(Δ::GroupAlgebraElement{T}, κ::T)
    return Δ*Δ - κ*Δ
end

function algebra_square(vector, elt)
    zzz = zeros(elt.coefficients)
    zzz[1:length(vector)] = vector
#     new_base_elt = GroupAlgebraElement(zzz, elt.product_matrix)
#     return (new_base_elt*new_base_elt).coefficients
    return GroupAlgebras.algebra_multiplication(zzz, zzz, elt.product_matrix)
end

function compute_SOS(sqrt_matrix, elt)
    n = size(sqrt_matrix,2)
    T = eltype(sqrt_matrix)

    # result = zeros(T, length(elt.coefficients))
    # for i in 1:n
    #     result += algebra_square(sqrt_matrix[:,i], elt)
    # end

    result = @parallel (+) for i in 1:n
        PropertyT.algebra_square(sqrt_matrix[:,i], elt)
    end

    return GroupAlgebraElement(result, elt.product_matrix)
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
        ɛ_dist = GroupAlgebras.ɛ(SOS_diff)
        @printf("ɛ(Δ² - κΔ - ∑ξᵢ*ξᵢ) ≈ %.10f\n", ɛ_dist)
        @printf("‖Δ² - κΔ - ∑ξᵢ*ξᵢ‖₁ ≈  %.10f\n", eoi_SOS_L₁_dist)
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
 
