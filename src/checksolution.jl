import Base: rationalize

using ValidatedNumerics
ValidatedNumerics.setrounding(Interval, :correct)
# ValidatedNumerics.setrounding(Interval, :fast) #which is slower??
ValidatedNumerics.setformat(:standard)
# setprecision(Interval, 53) # slightly faster than 256

function EOI{T<:Number}(Δ::GroupRingElem{T}, λ::T)
    return Δ*Δ - λ*Δ
end

function groupring_square(vect, elt)
    zzz = zeros(eltype(vect), elt.coeffs)
    zzz[1:length(vect)] = vect
    return GroupRings.groupring_mult(zzz, zzz, parent(elt).pm)
end

function compute_SOS(sqrt_matrix, elt)
    n = size(sqrt_matrix,2)
    T = eltype(sqrt_matrix)

    # result = zeros(T, length(elt.coeffs))
    # for i in 1:n
    #     result += groupring_square(sqrt_matrix[:,i], elt)
    # end

    result = @parallel (+) for i in 1:n
        groupring_square(sqrt_matrix[:,i], elt)
    end
    return GroupRingElem(result, parent(elt))
end

function correct_to_augmentation_ideal{T<:Rational}(sqrt_matrix::Array{T,2})
    sqrt_corrected = similar(sqrt_matrix)
    l = size(sqrt_matrix,2)
    for i in 1:l
        col = sqrt_matrix[:,i]
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

(±)(X::GroupRingElem, tol::Real) = GroupRingElem(X.coeffs ± tol, parent(X))

function Base.rationalize{T<:Integer, S<:Real}(::Type{T},
    X::AbstractArray{S}; tol::Real=eps(eltype(X)))
    r(x) = rationalize(T, x, tol=tol)
    return r.(X)
end

ℚ(x, tol::Real) = rationalize(BigInt, x, tol=tol)

function distance_to_cone{T<:Rational}(λ::T, sqrt_matrix::Array{T,2}, Δ::GroupRingElem{T}; len=4)
    SOS = compute_SOS(sqrt_matrix, Δ)

    SOS_diff = EOI(Δ, λ) - SOS
    eoi_SOS_L1_dist = norm(SOS_diff,1)

    info(logger, "λ = $λ (≈$(@sprintf("%.10f", float(λ)))")
    ɛ_dist = GroupRings.augmentation(SOS_diff)
    if ɛ_dist ≠ 0//1
        warn(logger, "The SOS is not in the augmentation ideal, numbers below are meaningless!")
    end
    info(logger, "ɛ(Δ² - λΔ - ∑ξᵢ*ξᵢ) = $ɛ_dist")
    info(logger, "‖Δ² - λΔ - ∑ξᵢ*ξᵢ‖₁ = $(@sprintf("%.10f", float(eoi_SOS_L1_dist)))")

    distance_to_cone = λ - 2^(len-1)*eoi_SOS_L1_dist
    return distance_to_cone
end

function distance_to_cone{T<:Rational, S<:Interval}(λ::T, sqrt_matrix::Array{S,2}, Δ::GroupRingElem{T}; len=4)
    SOS = compute_SOS(sqrt_matrix, Δ)
    info(logger, "ɛ(∑ξᵢ*ξᵢ) ∈ $(GroupRings.augmentation(SOS))")
    λ_int = @interval(λ)
    Δ_int = GroupRingElem([@interval(c) for c in Δ.coeffs], parent(Δ).pm)
    SOS_diff = EOI(Δ_int, λ_int) - SOS
    eoi_SOS_L1_dist = norm(SOS_diff,1)

    info(logger, "λ = $λ (≈≥$(@sprintf("%.10f",float(λ))))")
    ɛ_dist = GroupRings.augmentation(SOS_diff)

    info(logger, "ɛ(Δ² - λΔ - ∑ξᵢ*ξᵢ) ∈ $(ɛ_dist)")
    info(logger, "‖Δ² - λΔ - ∑ξᵢ*ξᵢ‖₁ ∈ $(eoi_SOS_L1_dist)")

    distance_to_cone = λ - 2^(len-1)*eoi_SOS_L₁_dist
    return distance_to_cone
end

function distance_to_cone{T<:AbstractFloat}(λ::T, sqrt_matrix::Array{T,2}, Δ::GroupRingElem{T}; len=4)
    SOS = compute_SOS(sqrt_matrix, Δ)

    SOS_diff = EOI(Δ, λ) - SOS
    eoi_SOS_L₁_dist = norm(SOS_diff,1)

    info(logger, "λ = $λ")
    ɛ_dist = GroupRings.augmentation(SOS_diff)
    info(logger, "ɛ(Δ² - λΔ - ∑ξᵢ*ξᵢ) ≈ $(@sprintf("%.10f", ɛ_dist))")
    info(logger, "‖Δ² - λΔ - ∑ξᵢ*ξᵢ‖₁ ≈ $(@sprintf("%.10f", eoi_SOS_L₁_dist))")

    distance_to_cone = λ - 2^(len-1)*eoi_SOS_L₁_dist
    return distance_to_cone
end

function check_distance_to_positive_cone(Δ::GroupRingElem, λ, P;
    tol=1e-7, rational=false, len=4)

    isapprox(eigvals(P), abs(eigvals(P)), atol=tol) ||
        warn("The solution matrix doesn't seem to be positive definite!")
    @assert P == Symmetric(P)
    Q = real(sqrtm(P))

    info(logger, "------------------------------------------------------------")
    info(logger, "")
    info(logger, "Checking in floating-point arithmetic...")
    t = @timed fp_distance = distance_to_cone(λ, Q, Δ, len=len)
    info(logger, timed_msg(t))
    info(logger, "Floating point distance (to positive cone) ≈ $(@sprintf("%.10f", fp_distance))")
    info(logger, "------------------------------------------------------------")

    if fp_distance ≤ 0
        return fp_distance
    end

    info(logger, "Projecting columns of rationalized Q to the augmentation ideal...")
    δ = eps(λ)
    Q_ℚ = ℚ(Q, δ)
    t = @timed Q_ℚω = correct_to_augmentation_ideal(Q_ℚ)
    info(logger, timed_msg(t))
    λ_ℚ = ℚ(λ, δ)
    Δ_ℚ = ℚ(Δ, δ)

    info(logger, "Checking in interval arithmetic")
    Q_ℚω_int = Float64.(Q_ℚω) ± δ
    t = @timed Interval_dist_to_Σ² = distance_to_cone(λ_ℚ, Q_ℚω_int, Δ_ℚ, len=len)
    info(logger, timed_msg(t))
    info(logger, "The Augmentation-projected actual distance (to positive cone) ∈ $(Interval_dist_to_Σ²)")
    info(logger, "------------------------------------------------------------")

    if Interval_dist_to_Σ².lo ≤ 0 || !rational
        return Interval_dist_to_Σ²
    else
        info(logger, "Checking Projected SOS decomposition in exact rational arithmetic...")
        t = @timed ℚ_dist_to_Σ² = distance_to_cone(λ_ℚ, Q_ℚω, Δ_ℚ, len=len)
        info(logger, timed_msg(t))
        @assert isa(ℚ_dist_to_Σ², Rational)
        info(logger, "Augmentation-projected rational distance (to positive cone) ≥ $(Float64(trunc(ℚ_dist_to_Σ²,8)))")
        info(logger, "------------------------------------------------------------")
        return ℚ_dist_to_Σ²
    end
end
