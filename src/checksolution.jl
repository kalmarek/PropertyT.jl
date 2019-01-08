using IntervalArithmetic

IntervalArithmetic.setrounding(Interval, :tight)
IntervalArithmetic.setformat(sigfigs=12)

function compute_SOS(pm::Array{I,2}, Q) where I<:Integer
    result = zeros(eltype(Q), maximum(pm));
    for i in 1:size(Q,2)
        GroupRings.fmac!(result, view(Q,:,i), view(Q,:,i), pm)
    end
    return result
end

function compute_SOS(RG::GroupRing, Q::AbstractArray)
    result = compute_SOS(RG.pm, Q)
    return GroupRingElem(result, RG)
end

function augIdproj(Q::AbstractArray{T,2}) where {T<:Real}
    result = zeros(Interval{T}, size(Q))
    l = size(Q, 2)
    Threads.@threads for j in 1:l
        col = sum(view(Q, :,j))/l
        for i in 1:size(Q, 1)
            result[i,j] = @interval(Q[i,j] - col)
        end
    end
    return result
end

    end
end
