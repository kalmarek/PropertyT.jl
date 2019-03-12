using IntervalArithmetic

IntervalArithmetic.setrounding(Interval, :tight)
IntervalArithmetic.setformat(sigfigs=12)

function compute_SOS(pm::AbstractMatrix{<:Integer}, Q::AbstractMatrix{<:Real})
    thr_count = Threads.nthreads()
    
    
    d, r = divrem(size(Q,2), thr_count)
    batch_result = [zeros(eltype(Q), maximum(pm)) for _ in 1:thr_count]
    
    Threads.@threads for k in 1:Threads.nthreads()
        for i in 1:d
            idx = d*(k-1)+i
            GroupRings.fmac!(batch_result[k], view(Q,:,idx), view(Q,:,idx), pm)
        end
    end
    
    result = sum(batch_result)
    for idx in thr_count*d+1:(thr_count*d + r)
        GroupRings.fmac!(result, view(Q,:,idx), view(Q,:,idx), pm)
    end

    return result
end

function compute_SOS(RG::GroupRing, Q::AbstractMatrix)
    result = compute_SOS(RG.pm, Q)
    return GroupRingElem(result, RG)
end

function augIdproj(Q::AbstractMatrix{<:Real})
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
