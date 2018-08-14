using IntervalArithmetic

IntervalArithmetic.setrounding(Interval, :tight)
IntervalArithmetic.setformat(sigfigs=12)

import IntervalArithmetic.±

function (±)(X::SparseVector, tol::Real)
    I, V = findnz(X)
    Vint = [v ± tol for v in V]
    return sparsevec(I, Vint)
end

function (±)(X::Array{T}, tol::Real) where {T<:AbstractFloat}
    result = zeros(Interval{Float64}, size(X)...)
    for i in eachindex(X)
        if X[i] != zero(T)
            result[i] = X[i] ± tol
        end
    end
    return result
end

(±)(X::GroupRingElem, tol::Real) = GroupRingElem(X.coeffs ± tol, parent(X))

EOI{T<:Number}(Δ::GroupRingElem{T}, λ::T) = Δ*Δ - λ*Δ

function groupring_square(pm, vect::AbstractVector)
    zzz = zeros(eltype(vect), maximum(pm))
    return GroupRings.mul!(zzz, vect, vect, pm)
end

function compute_SOS(pm::Array{I,2}, Q::AbstractArray) where I<:Integer

    # result = zeros(eltype(Q), maximum(pm))
    # r = similar(result)
    # for i in 1:size(Q,2)
    #    print(" $i")
    #    result += GroupRings.mul!(r, view(Q,:,i), view(Q,:,i), pm)
    # end

    @everywhere groupring_square = PropertyT.groupring_square

    result = @parallel (+) for i in 1:size(Q,2)
        groupring_square(pm, Q[:,i])
    end

    return result
end

function compute_SOS(RG::GroupRing, Q::AbstractArray)
    result = compute_SOS(RG.pm, Q)
    return RG(result)
end

function augIdproj(Q::AbstractArray{T,2}) where {T<:Real}
    R = zeros(Interval{T}, size(Q))
    l = size(Q, 2)
    Threads.@threads for j in 1:l
        col = sum(view(Q, :,j))/l
        for i in 1:size(Q, 1)
            R[i,j] = @interval(Q[i,j] - col)
        end
    end
    return R
end

function augIdproj(Q::AbstractArray{T,2}, logger) where {T<:Real}
    info(logger, "Projecting columns of Q to the augmentation ideal...")
    @logtime logger Q = augIdproj(Q)

    info(logger, "Checking that sum of every column contains 0.0... ")
    check = all([zero(T) in sum(view(Q, :, i)) for i in 1:size(Q, 2)])
    info(logger, (check? "They do." : "FAILED!"))

    @assert check

    return Q
end

function distance_to_cone(Δ::GroupRingElem, λ, Q; wlen::Int=4, logger=getlogger())
    info(logger, "------------------------------------------------------------")
    info(logger, "Checking in floating-point arithmetic...")
    info(logger, "λ = $λ")
    @logtime logger sos = compute_SOS(parent(Δ), Q)
    residue = Δ^2-λ*Δ - sos
    info(logger, "ɛ(Δ² - λΔ - ∑ξᵢ*ξᵢ) ≈ $(@sprintf("%.10f", aug(residue)))")
    L1_norm = norm(residue,1)
    info(logger, "‖Δ² - λΔ - ∑ξᵢ*ξᵢ‖₁ ≈ $(@sprintf("%.10f", L1_norm))")

    distance = λ - 2^(wlen-1)*L1_norm

    info(logger, "Floating point distance (to positive cone) ≈")
    info(logger, "$(@sprintf("%.10f", distance))")
    info(logger, "")

    if distance ≤ 0
        return distance
    end

    λ = @interval(λ)
    eoi = Δ^2 - λ*Δ
    Q = augIdproj(Q, logger)

    info(logger, "------------------------------------------------------------")
    info(logger, "Checking in interval arithmetic...")
    info(logger, "λ ∈ $λ")
    @logtime logger sos = compute_SOS(parent(Δ), Q)
    residue = Δ^2-λ*Δ - sos
    info(logger, "ɛ(∑ξᵢ*ξᵢ) ∈ $(aug(residue))")
    info(logger, "‖Δ² - λΔ - ∑ξᵢ*ξᵢ‖₁ ∈ $(L1_norm)")

    distance = λ - 2^(wlen-1)*L1_norm
    info(logger, "The Augmentation-projected distance (to positive cone) ∈")
    info(logger, "$(distance)")
    info(logger, "")

    return distance.lo
end
