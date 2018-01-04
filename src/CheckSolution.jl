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

EOI{T<:Number}(Δ::GroupRingElem{T}, λ::T) = Δ*Δ - λ*Δ

function groupring_square(vect::AbstractVector, l, pm)
    zzz = zeros(eltype(vect), l)
    return GroupRings.mul!(zzz, vect, vect, pm)
end

function compute_SOS(Q::AbstractArray, pm::Array{Int,2}, l::Int)

    # result = zeros(eltype(Q), l)
    # r = similar(result)
    # for i in 1:size(Q,2)
    #    print(" $i")
    #    result += GroupRings.mul!(r, view(Q,:,i), view(Q,:,i), pm)
    # end

    @everywhere groupring_square = PropertyT.groupring_square

    result = @parallel (+) for i in 1:size(Q,2)
        groupring_square(Q[:,i], l, pm)
    end

    return result

end

function compute_SOS(Q::AbstractArray, RG::GroupRing, l::Int)
    result = compute_SOS(Q, RG.pm, l)
    return GroupRingElem(result, RG)
end

function distances_to_cone(elt::GroupRingElem, wlen::Int)
    ɛ_dist = GroupRings.augmentation(elt)

    eoi_SOS_L1_dist = norm(elt,1)

    dist = 2^(wlen-1)*eoi_SOS_L1_dist
    return dist, ɛ_dist, eoi_SOS_L1_dist
end

function augIdproj{T, I<:AbstractInterval}(S::Type{I}, Q::AbstractArray{T,2})
    l = size(Q, 2)
    R = zeros(S, (l,l))
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
    @logtime logger Q = augIdproj(Interval{T}, Q)

    info(logger, "Checking that sum of every column contains 0.0... ")
    check = all([0.0 in sum(view(Q, :, i)) for i in 1:size(Q, 2)])
    info(logger, (check? "They do." : "FAILED!"))

    @assert check

    return Q
end

function distance_to_cone(elt::GroupRingElem, λ::T, Q::AbstractArray{T,2}, wlen::Int, logger) where {T<:AbstractFloat}

    info(logger, "------------------------------------------------------------")
    info(logger, "λ = $λ")
    info(logger, "Checking in floating-point arithmetic...")
    @logtime logger SOS_diff = elt - compute_SOS(Q, parent(elt), length(elt.coeffs))
    dist, ɛ_dist, eoi_SOS_L1_dist = distances_to_cone(SOS_diff, wlen)
    info(logger, "ɛ(Δ² - λΔ - ∑ξᵢ*ξᵢ) ≈ $(@sprintf("%.10f", ɛ_dist))")
    info(logger, "‖Δ² - λΔ - ∑ξᵢ*ξᵢ‖₁ ≈ $(@sprintf("%.10f", eoi_SOS_L1_dist))")

    fp_distance = λ - dist

    info(logger, "Floating point distance (to positive cone) ≈")
    info(logger, "$(@sprintf("%.10f", fp_distance))")
    info(logger, "")

    return fp_distance
end

function distance_to_cone(elt::GroupRingElem, λ::T, Q::AbstractArray{T,2}, wlen::Int, logger) where {T<:AbstractInterval}
    info(logger, "------------------------------------------------------------")
    info(logger, "λ = $λ")
    info(logger, "Checking in interval arithmetic...")
    @logtime logger SOS_diff = elt - compute_SOS(Q, parent(elt), length(elt.coeffs))
    dist, ɛ_dist, eoi_SOS_L1_dist = distances_to_cone(SOS_diff, wlen)
    info(logger, "ɛ(∑ξᵢ*ξᵢ) ∈ $(ɛ_dist)")
    info(logger, "‖Δ² - λΔ - ∑ξᵢ*ξᵢ‖₁ ∈ $(eoi_SOS_L1_dist)")

    int_distance = λ - dist

    info(logger, "The Augmentation-projected actual distance (to positive cone) ∈")
    info(logger, "$(int_distance)")
    info(logger, "")

    return int_distance
end

function check_distance_to_cone(Δ::GroupRingElem, λ, Q, wlen::Int, logger)

    fp_distance = distance_to_cone(EOI(Δ, λ), λ, Q, wlen, logger)

    if fp_distance ≤ 0
        return fp_distance
    end

    λ = @interval(λ)
    Δ = GroupRingElem([@interval(c) for c in Δ.coeffs], parent(Δ))
    Q = augIdproj(Q, logger)

    int_distance = distance_to_cone(EOI(Δ, λ), λ, Q, wlen, logger)

    return int_distance.lo
end
