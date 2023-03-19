module Roots

using StaticArrays
using LinearAlgebra

export Root, isproportional, isorthogonal, ~, ⟂

abstract type AbstractRoot{N,T} end

struct Root{N,T} <: AbstractRoot{N,T}
    coord::SVector{N,T}
end

Root(a) = Root(SVector(a...))

function Base.:(==)(r::Root{N}, s::Root{M}) where {M,N}
    M == N || return false
    r.coord == s.coord || return false
    return true
end

Base.hash(r::Root, h::UInt) = hash(r.coord, hash(Root, h))

Base.:+(r::Root{N,T}, s::Root{N,T}) where {N,T} = Root{N,T}(r.coord + s.coord)
Base.:-(r::Root{N,T}, s::Root{N,T}) where {N,T} = Root{N,T}(r.coord - s.coord)
Base.:-(r::Root{N}) where {N} = Root(-r.coord)

Base.:*(a::Number, r::Root) = Root(a * r.coord)
Base.:*(r::Root, a::Number) = a * r

Base.length(r::AbstractRoot) = norm(r, 2)

LinearAlgebra.norm(r::Root, p::Real = 2) = norm(r.coord, p)
LinearAlgebra.dot(r::Root, s::Root) = dot(r.coord, s.coord)

cos_angle(a, b) = dot(a, b) / (norm(a) * norm(b))

function isproportional(α::AbstractRoot{N}, β::AbstractRoot{M}) where {N,M}
    N == M || return false
    val = abs(cos_angle(α, β))
    return isapprox(val, one(val); atol = eps(one(val)))
end

function isorthogonal(α::AbstractRoot{N}, β::AbstractRoot{M}) where {N,M}
    N == M || return false
    val = cos_angle(α, β)
    return isapprox(val, zero(val); atol = eps(one(val)))
end

function _positive_direction(α::Root{N}) where {N}
    v = α.coord + 1 / (N * 100) * rand(N)
    return Root{N,Float64}(v / norm(v, 2))
end

function positive(roots::AbstractVector{<:Root{N}}) where {N}
    pd = _positive_direction(first(roots))
    return filter(α -> dot(α, pd) > 0.0, roots)
end

function Base.show(io::IO, r::Root)
    return print(io, "Root$(r.coord)")
end

function Base.show(io::IO, ::MIME"text/plain", r::Root{N}) where {N}
    lngth² = sum(x -> x^2, r.coord)
    l = isinteger(sqrt(lngth²)) ? "$(sqrt(lngth²))" : "√$(lngth²)"
    return print(io, "Root in ℝ^$N of length $l\n", r.coord)
end

𝕖(N, i) = Root(ntuple(k -> k == i ? 1 : 0, N))
𝕆(N, ::Type{T}) where {T} = Root(ntuple(_ -> zero(T), N))

reflection(α::Root, β::Root) = β - Int(2dot(α, β) / dot(α, α)) * α
function cartan(α, β)
    return [
        length(reflection(a, b) - b) / length(a) for a in (α, β), b in (α, β)
    ]
end

"""
    classify_root_system(α, β)
Return the symbol of smallest system generated by roots `α` and `β`.

The classification is based only on roots length,
proportionality/orthogonality and Cartan matrix.
"""
function classify_root_system(
    α::AbstractRoot,
    β::AbstractRoot,
    long::Tuple{Bool,Bool},
)
    if isproportional(α, β)
        if all(long)
            return :C₁
        elseif all(.!long) # both short
            return :A₁
        else
            @error "Proportional roots of different length"
            error("Unknown root system ⟨α, β⟩:\n α = $α\n β = $β")
        end
    elseif isorthogonal(α, β)
        if all(long)
            return Symbol("C₁×C₁")
        elseif all(.!long) # both short
            return Symbol("A₁×A₁")
        elseif any(long)
            return Symbol("A₁×C₁")
        end
    else # ⟨α, β⟩ is 2-dimensional, but they're not orthogonal
        a, b, c, d = abs.(cartan(α, β))
        @assert a == d == 2
        b, c = b < c ? (b, c) : (c, b)
        if b == c == 1
            return :A₂
        elseif b == 1 && c == 2
            return :C₂
        elseif b == 1 && c == 3
            @warn ":G₂? really?"
            return :G₂
        else
            @error a, b, c, d
            error("Unknown root system ⟨α, β⟩:\n α = $α\n β = $β")
        end
    end
end

function proportional_root_from_system(Ω::AbstractVector{<:Root}, α::Root)
    k = findfirst(v -> isproportional(α, v), Ω)
    if isnothing(k)
        error("Line L_α not contained in root system Ω:\n α = $α\n Ω = $Ω")
    end
    return Ω[k]
end

struct Plane{R<:Root}
    v1::R
    v2::R
    vectors::Vector{R}
end

function Plane(α::Root, β::Root)
    return Plane(α, β, [a * α + b * β for a in -3:3 for b in -3:3])
end

function Base.in(r::Root, plane::Plane)
    return any(isproportional(r, v) for v in plane.vectors)
end

function _islong(α::Root, Ω)
    lα = length(α)
    return any(r -> lα - length(r) > eps(lα), Ω)
end

function classify_sub_root_system(
    Ω::AbstractVector{<:Root{N}},
    α::Root{N},
    β::Root{N},
) where {N}
    @assert 1 ≤ length(unique(length, Ω)) ≤ 2
    v = proportional_root_from_system(Ω, α)
    w = proportional_root_from_system(Ω, β)

    subsystem = filter(ω -> ω in Plane(v, w), Ω)
    @assert length(subsystem) > 0
    subsystem = positive(union(subsystem, -1 .* subsystem))

    l = length(subsystem)
    if l == 1
        x = first(subsystem)
        long = _islong(x, Ω)
        return classify_root_system(x, -x, (long, long))
    elseif l == 2
        x, y = subsystem
        return classify_root_system(x, y, (_islong(x, Ω), _islong(y, Ω)))
    elseif l == 3
        x, y, z = subsystem
        l1, l2, l3 = _islong(x, Ω), _islong(y, Ω), _islong(z, Ω)
        a = classify_root_system(x, y, (l1, l2))
        b = classify_root_system(y, z, (l2, l3))
        c = classify_root_system(x, z, (l1, l3))

        if :A₂ == a == b == c # it's only A₂
            return a
        end

        throw("Unknown subroot system! $((x,y,z))")
    elseif l == 4
        subtypes = [
            classify_root_system(x, y, (_islong(x, Ω), _islong(y, Ω))) for
            x in subsystem for y in subsystem if x ≠ y
        ]
        if :C₂ in subtypes
            return :C₂
        end
    end
    @error "Unknown root subsystem generated by" α β
    throw("Unknown root system: $subsystem")
end

end # of module Roots
