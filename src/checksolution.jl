using IntervalArithmetic

IntervalArithmetic.setrounding(Interval, :tight)
IntervalArithmetic.setformat(sigfigs=12)

function fma_SOS_thr!(result::AbstractVector{T}, pm::AbstractMatrix{<:Integer},
    Q::AbstractMatrix{T}, acc_matrix=zeros(T, size(pm)...)) where T

    s1, s2 = size(pm)

    @inbounds for k in 1:s2
        let k=k, s1=s1, s2=s2, Q=Q, acc_matrix=acc_matrix
            Threads.@threads for j in 1:s2
                for i in 1:s1
                    @inbounds acc_matrix[i,j] = muladd(Q[i, k], Q[j, k], acc_matrix[i,j])
                end
            end
        end
    end

    @inbounds for j in 1:s2
        for i in 1:s1
            result[pm[i,j]] += acc_matrix[i,j]
        end
    end

    return result
end

function compute_SOS(pm::AbstractMatrix{<:Integer}, Q::AbstractMatrix)
    result = zeros(eltype(Q), maximum(pm));
    return fma_SOS_thr!(result, pm, Q)
end

function compute_SOS(RG::GroupRing, Q::AbstractMatrix{<:Real})
    result = compute_SOS(RG.pm, Q)
    return GroupRingElem(result, RG)
end

function compute_SOS_square(pm::AbstractMatrix{<:Integer}, Q::AbstractMatrix{<:Real})
    result = zeros(eltype(Q), maximum(pm));

    for i in 1:size(Q,2)
        GroupRings.fmac!(result, view(Q,:,i), view(Q,:,i), pm)
    end

    return result
end

function compute_SOS_square(RG::GroupRing, Q::AbstractMatrix{<:Real})
    return GroupRingElem(compute_SOS_square(RG.pm, Q), RG)
end

function augIdproj(Q::AbstractMatrix{T}) where {T<:Real}
    result = zeros(size(Q))
    l = size(Q, 2)
    Threads.@threads for j in 1:l
        col = sum(view(Q, :,j))/l
        for i in 1:size(Q, 1)
            result[i,j] = Q[i,j] - col
        end
    end
    return result
end

function augIdproj(::Type{Interval}, Q::AbstractMatrix{T}) where {T<:Real}
    result = zeros(Interval{T}, size(Q))
    l = size(Q, 2)
    Threads.@threads for j in 1:l
        col = sum(view(Q, :,j))/l
        for i in 1:size(Q, 1)
            result[i,j] = @interval(Q[i,j] - col)
        end
    end
    check = all([zero(T) in sum(view(result, :, i)) for i in 1:size(result, 2)])
    return result, check
end
