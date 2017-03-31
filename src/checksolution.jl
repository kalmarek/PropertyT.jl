import Base: rationalize

using ValidatedNumerics
ValidatedNumerics.setrounding(Interval, :correct)
ValidatedNumerics.setformat(:standard)

function EOI{T<:Number}(Δ::GroupAlgebraElement{T}, κ::T)
    return Δ*Δ - κ*Δ
end

function algebra_square(vector, elt)
    zzz = zeros(eltype(vector), elt.coefficients)
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

import ValidatedNumerics.±

function (±){T<:Number}(X::AbstractArray{T}, tol::Real)
    r{T}(x::T) = (x == zero(T)? @interval(0) : x ± tol)
    return r.(X)
end

(±)(X::GroupAlgebraElement, tol::Real) = GroupAlgebraElement(X.coefficients ± tol, X.product_matrix)

function Base.rationalize{T<:Integer, S<:Real}(::Type{T},
    X::AbstractArray{S}; tol::Real=eps(eltype(X)))
    r(x) = rationalize(T, x, tol=tol)
    return r.(X)
end

ℚ(x, tol::Real) = rationalize(BigInt, x, tol=tol)

function distance_to_cone{T<:Rational}(κ::T, sqrt_matrix::Array{T,2}, Δ::GroupAlgebraElement{T})
    SOS = compute_SOS(sqrt_matrix, Δ)

    SOS_diff = EOI(Δ, κ) - SOS
    eoi_SOS_L₁_dist = norm(SOS_diff,1)

    info(logger, "κ = $κ (≈$(@sprintf("%.10f", float(κ)))")
    ɛ_dist = GroupAlgebras.ɛ(SOS_diff)
    if ɛ_dist ≠ 0//1
        warn(logger, "The SOS is not in the augmentation ideal, number below are meaningless!")
    end
    info(logger, "ɛ(Δ² - κΔ - ∑ξᵢ*ξᵢ) = $ɛ_dist")
    info(logger, "‖Δ² - κΔ - ∑ξᵢ*ξᵢ‖₁ = $(@sprintf("%.10f", float(eoi_SOS_L₁_dist)))")

    distance_to_cone = κ - 2^3*eoi_SOS_L₁_dist
    return distance_to_cone
end

function distance_to_cone{T<:Rational, S<:Interval}(κ::T, sqrt_matrix::Array{S,2}, Δ::GroupAlgebraElement{T})
    SOS = compute_SOS(sqrt_matrix, Δ)
    info(logger, "ɛ(∑ξᵢ*ξᵢ) ∈ $(GroupAlgebras.ɛ(SOS))")

    SOS_diff = EOI(Δ, κ) - SOS
    eoi_SOS_L₁_dist = norm(SOS_diff,1)

    info(logger, "κ = $κ (≈$(@sprintf("%.10f",float(κ))))")
    ɛ_dist = GroupAlgebras.ɛ(SOS_diff)

    info(logger, "ɛ(Δ² - κΔ - ∑ξᵢ*ξᵢ) ∈ $(ɛ_dist)")
    info(logger, "‖Δ² - κΔ - ∑ξᵢ*ξᵢ‖₁ ∈ $(eoi_SOS_L₁_dist)")

    distance_to_cone = κ - 2^3*eoi_SOS_L₁_dist
    return distance_to_cone
end

function distance_to_cone{T<:AbstractFloat}(κ::T, sqrt_matrix::Array{T,2}, Δ::GroupAlgebraElement{T})
    SOS = compute_SOS(sqrt_matrix, Δ)

    SOS_diff = EOI(Δ, κ) - SOS
    eoi_SOS_L₁_dist = norm(SOS_diff,1)

    info(logger, "κ = $κ")
    ɛ_dist = GroupAlgebras.ɛ(SOS_diff)
    info(logger, "ɛ(Δ² - κΔ - ∑ξᵢ*ξᵢ) ≈ $(@sprintf("%.10f", ɛ_dist))")
    info(logger, "‖Δ² - κΔ - ∑ξᵢ*ξᵢ‖₁ ≈ $(@sprintf("%.10f", eoi_SOS_L₁_dist))")

    distance_to_cone = κ - 2^3*eoi_SOS_L₁_dist
    return distance_to_cone
end

function check_distance_to_positive_cone(Δ::GroupAlgebraElement, κ, A;
    tol=1e-7, rational=false)

    isapprox(eigvals(A), abs(eigvals(A)), atol=tol) ||
        warn("The solution matrix doesn't seem to be positive definite!")
    @assert A == Symmetric(A)
    A_sqrt = real(sqrtm(A))

    info(logger, "------------------------------------------------------------")
    info(logger, "")
    info(logger, "Checking in floating-point arithmetic...")
    t = @timed fp_distance = distance_to_cone(κ, A_sqrt, Δ)
    info(logger, timed_msg(t))
    info(logger, "Floating point distance (to positive cone) ≈ $(@sprintf("%.10f", fp_distance))")
    info(logger, "------------------------------------------------------------")

    info(logger, "Projecting columns of rationalized A_sqrt to the augmentation ideal...")
    δ = eps(κ)
    A_sqrt_ℚ = ℚ(A_sqrt, δ)
    t = @timed A_sqrt_ℚ_aug = correct_to_augmentation_ideal(A_sqrt_ℚ)
    info(logger, timed_msg(t))
    κ_ℚ = ℚ(κ, δ)
    Δ_ℚ = ℚ(Δ, δ)

    info(logger, "Checking in interval arithmetic")
    A_sqrt_ℚ_augⁱⁿᵗ = Float64.(A_sqrt_ℚ_aug) ± δ
    t = @timed Interval_dist_to_Σ² = distance_to_cone(κ_ℚ, A_sqrt_ℚ_augⁱⁿᵗ, Δ_ℚ)
    info(logger, timed_msg(t))
    info(logger, "The Augmentation-projected actual distance (to positive cone) ≥ $(@sprintf("%.10f", Interval_dist_to_Σ².lo))")
    info(logger, "------------------------------------------------------------")

    if Interval_dist_to_Σ².lo ≤ 0 || !rational
        return Interval_dist_to_Σ².lo
    else
        info(logger, "Checking Projected SOS decomposition in exact rational arithmetic...")
        t = @timed ℚ_dist_to_Σ² = distance_to_cone(κ_ℚ, A_sqrt_ℚ_aug, Δ_ℚ)
        info(logger, timed_msg(t))
        @assert isa(ℚ_dist_to_Σ², Rational)
        info(logger, "Augmentation-projected rational distance (to positive cone) ≥ $(Float64(trunc(ℚ_dist_to_Σ²,8)))")
        info(logger, "------------------------------------------------------------")
        return ℚ_dist_to_Σ²
    end
end
