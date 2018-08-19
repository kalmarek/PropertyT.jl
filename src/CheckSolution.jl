using IntervalArithmetic

IntervalArithmetic.setrounding(Interval, :tight)
IntervalArithmetic.setformat(sigfigs=12)

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
    return GroupRingElem(result, RG)
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

function distance_to_cone(Δ::GroupRingElem, λ, Q; wlen::Int=4)
    info("------------------------------------------------------------")
    info("Checking in floating-point arithmetic...")
    info("λ = $λ")
    @time sos = compute_SOS(parent(Δ), Q)
    residue = Δ^2-λ*Δ - sos
    info("ɛ(Δ² - λΔ - ∑ξᵢ*ξᵢ) ≈ $(@sprintf("%.10f", aug(residue)))")
    L1_norm = norm(residue,1)
    info("‖Δ² - λΔ - ∑ξᵢ*ξᵢ‖₁ ≈ $(@sprintf("%.10f", L1_norm))")

    distance = λ - 2^(wlen-1)*L1_norm

    info("Floating point distance (to positive cone) ≈")
    info("$(@sprintf("%.10f", distance))")
    info("")

    if distance ≤ 0
        return distance
    end

    info("------------------------------------------------------------")
    info("Checking in interval arithmetic...")
    info("λ ∈ $λ")

    λ = @interval(λ)
    eoi = Δ^2 - λ*Δ
    info("Projecting columns of Q to the augmentation ideal...")
    T = eltype(Q)
    @time Q = augIdproj(Q)

    info("Checking that sum of every column contains 0.0... ")
    check = all([zero(T) in sum(view(Q, :, i)) for i in 1:size(Q, 2)])
    info((check? "They do." : "FAILED!"))

    @assert check

    @time sos = compute_SOS(parent(Δ), Q)
    residue = Δ^2-λ*Δ - sos
    info("ɛ(∑ξᵢ*ξᵢ) ∈ $(aug(residue))")
    L1_norm = norm(residue,1)
    info("‖Δ² - λΔ - ∑ξᵢ*ξᵢ‖₁ ∈ $(L1_norm)")

    distance = λ - 2^(wlen-1)*L1_norm
    info("The Augmentation-projected distance (to positive cone) ∈")
    info("$(distance)")
    info("")

    return distance.lo
end
