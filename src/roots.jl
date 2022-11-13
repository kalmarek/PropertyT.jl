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

LinearAlgebra.norm(r::Root, p::Real=2) = norm(r.coord, p)
LinearAlgebra.dot(r::Root, s::Root) = dot(r.coord, s.coord)

cos_angle(a, b) = dot(a, b) / (norm(a) * norm(b))

function isproportional(α::AbstractRoot{N}, β::AbstractRoot{M}) where {N,M}
    N == M || return false
    val = abs(cos_angle(α, β))
    return isapprox(val, one(val), atol=eps(one(val)))
end

function isorthogonal(α::AbstractRoot{N}, β::AbstractRoot{M}) where {N,M}
    N == M || return false
    val = cos_angle(α, β)
    return isapprox(val, zero(val), atol=eps(one(val)))
end

function _positive_direction(α::Root{N}) where {N}
    v = α.coord + 1 / (N * 100) * rand(N)
    return Root{N,Float64}(v / norm(v, 2))
end

function positive(roots::AbstractVector{<:Root{N}}) where {N}
    # return those roots for which dot(α, Root([½, ¼, …])) > 0.0
    pd = _positive_direction(first(roots))
    return filter(α -> dot(α, pd) > 0.0, roots)
end

function Base.show(io::IO, r::Root)
    print(io, "Root$(r.coord)")
end

function Base.show(io::IO, ::MIME"text/plain", r::Root{N}) where {N}
    lngth² = sum(x -> x^2, r.coord)
    l = isinteger(sqrt(lngth²)) ? "$(sqrt(lngth²))" : "√$(lngth²)"
    print(io, "Root in ℝ^$N of length $l\n", r.coord)
end

𝕖(N, i) = Root(ntuple(k -> k == i ? 1 : 0, N))
𝕆(N, ::Type{T}) where {T} = Root(ntuple(_ -> zero(T), N))

"""
    classify_root_system(α, β)
Return the symbol of smallest system generated by roots `α` and `β`.

The classification is based only on roots length and
proportionality/orthogonality.
"""
function classify_root_system(α::AbstractRoot, β::AbstractRoot)
    lα, lβ = length(α), length(β)
    if isproportional(α, β)
        if lα ≈ lβ ≈ √2
            return :A₁
        elseif lα ≈ lβ ≈ 2.0
            return :C₁
        else
            error("Unknown root system ⟨α, β⟩:\n α = $α\n β = $β")
        end
    elseif isorthogonal(α, β)
        if lα ≈ lβ ≈ √2
            return Symbol("A₁×A₁")
        elseif lα ≈ lβ ≈ 2.0
            return Symbol("C₁×C₁")
        elseif (lα ≈ 2.0 && lβ ≈ √2) || (lα ≈ √2 && lβ ≈ 2)
            return Symbol("A₁×C₁")
        else
            error("Unknown root system ⟨α, β⟩:\n α = $α\n β = $β")
        end
    else # ⟨α, β⟩ is 2-dimensional, but they're not orthogonal
        if lα ≈ lβ ≈ √2
            return :A₂
        elseif (lα ≈ 2.0 && lβ ≈ √2) || (lα ≈ √2 && lβ ≈ 2)
            return :C₂
        else
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

Plane(α::R, β::R) where {R<:Root} =
    Plane(α, β, [a * α + b * β for a in -3:3 for b in -3:3])

function Base.in(r::R, plane::Plane{R}) where {R}
    return any(isproportional(r, v) for v in plane.vectors)
end

function classify_sub_root_system(
    Ω::AbstractVector{<:Root{N}},
    α::Root{N},
    β::Root{N},
) where {N}

    v = proportional_root_from_system(Ω, α)
    w = proportional_root_from_system(Ω, β)

    subsystem = filter(ω -> ω in Plane(v, w), Ω)
    @assert length(subsystem) > 0
    subsystem = positive(union(subsystem, -1 .* subsystem))

    l = length(subsystem)
    if l == 1
        x = first(subsystem)
        return classify_root_system(x, x)
    elseif l == 2
        return classify_root_system(subsystem...)
    elseif l == 3
        a = classify_root_system(subsystem[1], subsystem[2])
        b = classify_root_system(subsystem[2], subsystem[3])
        c = classify_root_system(subsystem[1], subsystem[3])

        if a == b == c # it's only A₂
            return a
        end

        C = (:C₂, Symbol("C₁×C₁"))
        if (a ∈ C && b ∈ C && c ∈ C) && (:C₂ ∈ (a, b, c))
            return :C₂
        end
    elseif l == 4
        for i = 1:l
            for j = (i+1):l
                T = classify_root_system(subsystem[i], subsystem[j])
                T == :C₂ && return :C₂
            end
        end
    end
    @error "Unknown root subsystem generated by" α β
    throw("Unknown root system: $subsystem")
end

end # of module Roots
