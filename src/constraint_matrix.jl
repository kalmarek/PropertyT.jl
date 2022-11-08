"""
    ConstraintMatrix{T,I} <: AbstractMatrix{T}

Special type of sparse matrix used to store constraints in SOS problems.

This matrix has in general very few non-zero values which also are multiples of each other.

The constructor accepts
* `nzeros` - a vector of non-zero indices; negative values are used to signify
 negative values; repetitions are allowed
* `n`, `m` - the size of matrix
* `val` - the greatest common factor of the values

To iterate efficiently over `A::ConstraintMatrix` use [`nzpairs(A)`](@ref).

# Examples
```julia-repl
julia> ConstraintMatrix{Float64}([-1,2,-1,1,4,2,6], 3,2, π)
3×2 ConstraintMatrix{Float64, Int64}:
 -3.14159  3.14159
  6.28319  0.0
  0.0      3.14159
```
"""
struct ConstraintMatrix{T,I} <: AbstractMatrix{T}
    pos::Vector{I} # list of positive indices
    neg::Vector{I} # list of negative indices
    size::Tuple{Int,Int}
    val::T

    function ConstraintMatrix{T}(nzeros::AbstractArray{<:Integer}, n, m, val) where {T}
        @assert n ≥ 1
        @assert m ≥ 1

        if !isempty(nzeros)
            sort!(nzeros)
            a, b = first(nzeros), last(nzeros)
            @assert 1 ≤ abs(a) ≤ n * m
            @assert 1 ≤ abs(b) ≤ n * m
        end
        k = searchsortedlast(nzeros, 0)
        neg = @view nzeros[begin:k]
        pos = @view nzeros[k+1:end]
        return new{T,eltype(nzeros)}(pos, -neg, (n, m), val)
    end
end

ConstraintMatrix(nzeros::AbstractArray{<:Integer}, n, m, val::T) where {T} =
    ConstraintMatrix{T}(nzeros, n, m, val)

Base.size(cm::ConstraintMatrix) = cm.size

__get_positive(cm::ConstraintMatrix, idx::Integer) =
    convert(eltype(cm), cm.val * length(searchsorted(cm.pos, idx)))
__get_negative(cm::ConstraintMatrix, idx::Integer) =
    convert(eltype(cm), cm.val * length(searchsorted(cm.neg, idx)))

Base.@propagate_inbounds function Base.getindex(
    cm::ConstraintMatrix,
    i::Integer,
    j::Integer,
)
    li = LinearIndices(cm)
    idx = li[i, j]
    pos = __get_positive(cm, idx)
    neg = __get_negative(cm, idx)
    return pos - neg
end

struct NZPairsIter{T}
    m::ConstraintMatrix{T}
end

Base.eltype(::Type{NZPairsIter{T}}) where {T} = Pair{Int,T}
Base.IteratorSize(::Type{<:NZPairsIter}) = Base.SizeUnknown()

# TODO: iterate over (idx=>val) pairs combining vals
function Base.iterate(itr::NZPairsIter, state::Tuple{Int,Int}=(1, 1))
    k = iterate(itr.m.pos, state[1])
    isnothing(k) && return iterate(itr, state[2])
    idx, st = k
    return idx => itr.m.val, (st, 1)
end

function Base.iterate(itr::NZPairsIter, state::Int)
    k = iterate(itr.m.neg, state[1])
    isnothing(k) && return nothing
    idx, st = k
    return idx => -itr.m.val, st
end

"""
    nzpairs(cm::ConstraintMatrix)
Efficiently iterate over non-zero `(idx=>value)` pairs.

If the `cm` was created with repetitions (or contains negative values) there will
be repetitions in the returned sequence of pairs.

# Examples
```julia
julia> ConstraintMatrix{Float64}([-1,2,-1,1,4,2,6], 3,2, π)
3×2 ConstraintMatrix{Float64, Int64}:
 -3.14159  3.14159
  6.28319  0.0
  0.0      3.14159

julia> collect(nzpairs(M))
7-element Vector{Pair{Int64, Float64}}:
 1 => 3.141592653589793
 2 => 3.141592653589793
 2 => 3.141592653589793
 4 => 3.141592653589793
 6 => 3.141592653589793
 1 => -3.141592653589793
 1 => -3.141592653589793
```
"""
nzpairs(cm::ConstraintMatrix) = NZPairsIter(cm)

function LinearAlgebra.dot(cm::ConstraintMatrix, m::AbstractMatrix{T}) where {T}
    if isempty(cm.pos) && isempty(cm.neg)
        isempty(m) && return zero(T)
        return zero(first(m) + first(m))
    end

    pos = isempty(cm.pos) ? zero(first(m)) : sum(@view m[cm.pos])
    neg = isempty(cm.neg) ? zero(first(m)) : sum(@view m[cm.neg])
    return convert(eltype(cm), cm.val) * (pos - neg)
end
