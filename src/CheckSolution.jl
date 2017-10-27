import Base: rationalize

using IntervalArithmetic

IntervalArithmetic.setrounding(Interval, :tight)
IntervalArithmetic.setformat(sigfigs=12)

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

function compute_SOS(Q::AbstractArray, pm::Array{Int,2}, l::Int)
   n = size(Q,2)

   # result = zeros(eltype(Q), l)
   # for i in 1:n
   #    result .+= groupring_square(view(Q,:,i), l, pm)
   # end

   @everywhere groupring_square = PropertyT.groupring_square

   result = @parallel (+) for i in 1:n
      print(" $i")
      groupring_square(view(Q,:,i), l, pm)
   end

   return result
end

function compute_SOS(Q::AbstractArray, RG::GroupRing, l::Int)
   result = compute_SOS(Q, RG.pm, l)
   return GroupRingElem(result, RG)
end

function distance_to_cone{S<:Interval}(elt::GroupRingElem, Q::AbstractArray{S,2}, wlen::Int)
   SOS = compute_SOS(Q, parent(elt), length(elt.coeffs))
   SOS_diff = elt - SOS

   ɛ_dist = GroupRings.augmentation(SOS_diff)
   info(logger, "ɛ(∑ξᵢ*ξᵢ) ∈ $(ɛ_dist)")

   eoi_SOS_L1_dist = norm(SOS_diff,1)
   info(logger, "‖Δ² - λΔ - ∑ξᵢ*ξᵢ‖₁ ∈ $(eoi_SOS_L1_dist)")

   dist = 2^(wlen-1)*eoi_SOS_L1_dist
   return dist
end

function distance_to_cone{T}(elt::GroupRingElem, Q::AbstractArray{T,2}, wlen::Int)
   SOS = compute_SOS(Q, parent(elt), length(elt.coeffs))
   SOS_diff = elt - SOS

   ɛ_dist = GroupRings.augmentation(SOS_diff)
   info(logger, "ɛ(Δ² - λΔ - ∑ξᵢ*ξᵢ) ≈ $(@sprintf("%.10f", ɛ_dist))")

   eoi_SOS_L1_dist = norm(SOS_diff,1)
   info(logger, "‖Δ² - λΔ - ∑ξᵢ*ξᵢ‖₁ ≈ $(@sprintf("%.10f", eoi_SOS_L1_dist))")

   dist = 2^(wlen-1)*eoi_SOS_L1_dist
   return dist
end

function augIdproj{T, I<:AbstractInterval}(S::Type{I}, Q::AbstractArray{T,2})
   l = size(Q, 2)
   R = zeros(Interval, (l,l))
   Threads.@threads for j in 1:l
      col = sum(view(Q, :,j))/l
      for i in 1:l
         R[i,j] = Q[i,j] - col ± eps(0.0)
      end
   end
   return R
end

function augIdproj{T}(Q::AbstractArray{T,2}, logger)
   info(logger, "Projecting columns of Q to the augmentation ideal...")
   @logtime logger Q = augIdproj(Interval, Q)

   info(logger, "Checking that sum of every column contains 0.0... ")
   check = all([0.0 in sum(view(Q, :, i)) for i in 1:size(Q, 2)])
   info(logger, (check? "They do." : "FAILED!"))

   @assert check

   return Q
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
    Q_ℚω_int = rationalize_and_project(Q, tol, logger)
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
