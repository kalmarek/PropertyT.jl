import Base: rationalize

using IntervalArithmetic

IntervalArithmetic.setrounding(Interval, :tight)
IntervalArithmetic.setformat(sigfigs=10)
IntervalArithmetic.setprecision(Interval, 53) # slightly faster than 256

import IntervalArithmetic.±

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

EOI{T<:Number}(Δ::GroupRingElem{T}, λ::T) = Δ*Δ - λ*Δ

function groupring_square(vect::AbstractVector, l, pm)
    zzz = zeros(eltype(vect), l)
    zzz[1:length(vect)] .= vect
    return GroupRings.mul!(similar(zzz), zzz, zzz, pm)
end

function compute_SOS(sqrt_matrix, elt::GroupRingElem)
   n = size(sqrt_matrix,2)
   l = length(elt.coeffs)
   pm = parent(elt).pm

   result = zeros(eltype(sqrt_matrix), l)
   for i in 1:n
      result .+= groupring_square(view(sqrt_matrix,:,i), l, pm)
   end

   # @everywhere groupring_square = PropertyT.groupring_square
   #
   # result = @parallel (+) for i in 1:n
   #    groupring_square(view(sqrt_matrix,:,i), length(elt.coeffs), parent(elt).pm)
   # end

   return GroupRingElem(result, parent(elt))
end

function correct_to_augmentation_ideal{T<:Rational}(sqrt_matrix::Array{T,2})
   l = size(sqrt_matrix, 2)
   sqrt_corrected = Array{Interval{Float64}}(l,l)
   Threads.@threads for j in 1:l
      col = sum(view(sqrt_matrix, :,j))//l
      for i in 1:l
         sqrt_corrected[i,j] = (Float64(sqrt_matrix[i,j]) - Float64(col)) ± eps(0.0)
      end
   end
   return sqrt_corrected
end

function distance_to_cone{T<:Rational}(λ::T, sqrt_matrix::Array{T,2}, Δ::GroupRingElem{T}, wlen)
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

    distance_to_cone = λ - 2^(wlen-1)*eoi_SOS_L1_dist
    return distance_to_cone
end

function distance_to_cone{T<:Rational, S<:Interval}(λ::T, sqrt_matrix::AbstractArray{S,2}, Δ::GroupRingElem{T}, wlen)
    SOS = compute_SOS(sqrt_matrix, Δ)
    info(logger, "ɛ(∑ξᵢ*ξᵢ) ∈ $(GroupRings.augmentation(SOS))")
    λ_int = @interval(λ)
    Δ_int = GroupRingElem([@interval(c) for c in Δ.coeffs], parent(Δ))
    SOS_diff = EOI(Δ_int, λ_int) - SOS
    eoi_SOS_L1_dist = norm(SOS_diff,1)

    info(logger, "λ = $λ (≈≥$(@sprintf("%.10f",float(λ))))")
    ɛ_dist = GroupRings.augmentation(SOS_diff)

    info(logger, "ɛ(Δ² - λΔ - ∑ξᵢ*ξᵢ) ∈ $(ɛ_dist)")
    info(logger, "‖Δ² - λΔ - ∑ξᵢ*ξᵢ‖₁ ∈ $(eoi_SOS_L1_dist)")

    distance_to_cone = λ - 2^(wlen-1)*eoi_SOS_L1_dist
    return distance_to_cone
end

function distance_to_cone(λ, sqrt_matrix::AbstractArray, Δ::GroupRingElem, wlen)
    SOS = compute_SOS(sqrt_matrix, Δ)

    SOS_diff = EOI(Δ, λ) - SOS
    eoi_SOS_L1_dist = norm(SOS_diff,1)

    info(logger, "λ = $λ")
    ɛ_dist = GroupRings.augmentation(SOS_diff)
    info(logger, "ɛ(Δ² - λΔ - ∑ξᵢ*ξᵢ) ≈ $(@sprintf("%.10f", ɛ_dist))")
    info(logger, "‖Δ² - λΔ - ∑ξᵢ*ξᵢ‖₁ ≈ $(@sprintf("%.10f", eoi_SOS_L1_dist))")

    distance_to_cone = λ - 2^(wlen-1)*eoi_SOS_L1_dist
    return distance_to_cone
end

function rationalize_and_project{T}(Q::AbstractArray{T}, δ::T, logger)
   info(logger, "")
   info(logger, "Rationalizing with accuracy $δ")
   t = @timed Q_ℚ = ℚ(Q, δ)
   info(logger, timed_msg(t))

   info(logger, "Projecting columns of the rationalized Q to the augmentation ideal...")
   t = @timed Q_int = correct_to_augmentation_ideal(Q_ℚ)
   info(logger, timed_msg(t))

   info(logger, "Checking that sum of every column contains 0.0... ")
   check = all([0.0 in sum(view(Q_int, :, i)) for i in 1:size(Q_int, 2)])
   info(logger, (check? "They do." : "FAILED!"))

   @assert check

   return Q_int
end

function check_distance_to_positive_cone(Δ::GroupRingElem, λ, Q, wlen;
    tol=1e-14, rational=false)

    info(logger, "------------------------------------------------------------")
    info(logger, "")
    info(logger, "Checking in floating-point arithmetic...")
    t = @timed fp_distance = distance_to_cone(λ, Q, Δ, wlen)
    info(logger, timed_msg(t))
    info(logger, "Floating point distance (to positive cone) ≈ $(@sprintf("%.10f", fp_distance))")
    info(logger, "------------------------------------------------------------")

    if fp_distance ≤ 0
        return fp_distance
    end

    info(logger, "")
    info(logger, "Projecting columns of the rationalized Q to the augmentation ideal...")
    Q_ℚω_int = rationalize_and_project(Q, max(tol, 1e-12), logger)
    λ_ℚ = ℚ(λ, tol)
    Δ_ℚ = ℚ(Δ, tol)

    info(logger, "Checking in interval arithmetic")

    t = @timed Interval_dist_to_ΣSq = distance_to_cone(λ_ℚ, Q_ℚω_int, Δ_ℚ, wlen)
    info(logger, timed_msg(t))
    info(logger, "The Augmentation-projected actual distance (to positive cone) ∈ $(Interval_dist_to_ΣSq)")
    info(logger, "------------------------------------------------------------")

    if Interval_dist_to_ΣSq.lo ≤ 0 || !rational
        return Interval_dist_to_ΣSq
    else
        info(logger, "Checking Projected SOS decomposition in exact rational arithmetic...")
        t = @timed ℚ_dist_to_ΣSq = distance_to_cone(λ_ℚ, Q_ℚω, Δ_ℚ, wlen)
        info(logger, timed_msg(t))
        @assert isa(ℚ_dist_to_ΣSq, Rational)
        info(logger, "Augmentation-projected rational distance (to positive cone) ≥ $(Float64(trunc(ℚ_dist_to_ΣSq,8)))")
        info(logger, "------------------------------------------------------------")
        return ℚ_dist_to_ΣSq
    end
end
